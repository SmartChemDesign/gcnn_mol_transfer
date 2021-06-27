import tensorflow as tf
import math
import numpy as np
import os
from rdkit import Chem
from deepchem.data.data_loader import featurize_smiles_np
from deepchem.feat import ConvMolFeaturizer, AdjacencyFingerprint
from deepchem.data import NumpyDataset, DiskDataset
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
import json
import pandas as pd
from itertools import chain
from keras import backend
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, log_loss, \
    recall_score, matthews_corrcoef, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import compute_class_weight, shuffle
import sys


CLASS_METRICS = {"confusion_matrix": confusion_matrix, "f1_score": f1_score, "roc_auc_score": roc_auc_score,
                 "precision_score": precision_score, "recall_score": recall_score,
                 "matthews_corrcoef": matthews_corrcoef, "accuracy_score": accuracy_score}

REG_METRICS = {"r2_score": r2_score, "mean_squared_error": mean_squared_error,
               "mean_absolute_error": mean_absolute_error}

GRAPH_CONV_REGRESSOR = "Graph convolutional regressor"
GRAPH_CONV_CLASSIFIER = "Graph convolutional classifier"

HYPERPARAMETERS = {"classification": {"n_tasks": 1, "graph_conv_layers": [128, 64], "dense_layer_size": 128,
                                      "dropout": 0.0, "mode": "classification", "number_atom_features": 75,
                                      "n_classes": 2, "batch_size": 100, "num_dense": 3, "learning_rate": 0.001,
                                      "transfer": False},
                   "regression": {"n_tasks": 1, "graph_conv_layers": [128, 64], "dense_layer_size": 128,
                                  "dropout": 0.0, "mode": "regression", "number_atom_features": 75,
                                  "uncertainty": False, "learning_rate": 0.001,
                                  "batch_size": 100, "num_dense": 3, "transfer": False}}


class TransferTrainer:
    """
    Class for managing training process
    """

    def __init__(self, model_class, output_folder, mode="classification",
                 n_split=5, frac_train=0.8, n_epochs=200, batch_size=100,
                 parameters=None, es_steps=5, reduce_lr=False):

        self.path_to_data = None
        self.valuename = None
        self.object_type = None
        self.output_folder = output_folder
        self.main_folder = None
        self.n_split = n_split
        self.frac_train = frac_train
        self.es_steps = es_steps
        self.batch_size = batch_size
        self.mode = mode
        self.dataset = None
        self.scaler = None
        self.train_set = None
        self.test_set = None
        self.train_valid_split = []
        self.model_class = model_class
        self.restore_folder = None
        self.layers_to_freeze = None
        self.models = []
        self.structure_filename = "net_structure_data.json"
        self.n_epochs = n_epochs
        self.best_restore_fold = None
        self.reduce_lr = reduce_lr
        self.featurize_threads = 1

        if parameters is None:
            self.parameters = HYPERPARAMETERS[self.mode]
        else:
            self.parameters = parameters
        self.lr = parameters["learning_rate"]

    def restore_model_params(self, path_to_folder, layers_to_freeze=None):
        if layers_to_freeze is None:
            layers_to_freeze = ["graph_conv"]
        self.layers_to_freeze = layers_to_freeze

        self.restore_folder = path_to_folder
        self.parameters["transfer"] = True
        self.restore_model_structure()  # change parameters dict with values from json file in pretrained folder
        self.find_best_fold()

    def restore_model_structure(self):
        path_to_str_json = os.path.join(self.restore_folder,
                                        self.structure_filename)  # TODO fix for auto best fold search maybe
        with open(path_to_str_json) as jf:
            structure_data = json.load(jf)
        self.parameters["graph_conv_layers"] = structure_data["graph_conv_layers"]
        self.parameters["dense_layer_size"] = structure_data["dense_layer_size"]
        self.parameters["num_dense"] = structure_data["num_dense"]

    def write_model_structure(self):
        path_to_str_json = os.path.join(self.main_folder, self.structure_filename)
        structure = {"graph_conv_layers": self.parameters["graph_conv_layers"],
                     "dense_layer_size": self.parameters["dense_layer_size"],
                     "num_dense": self.parameters["num_dense"]
                     }

        with open(path_to_str_json, "w") as jf:
            json.dump(structure, jf)

    def prepare_out_folder(self):
        time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
        dataset_name = os.path.basename(self.path_to_data).split(".")[0]
        self.main_folder = os.path.join(self.output_folder, f"{self.mode}_{dataset_name}_{self.valuename}_{time_mark}")
        os.mkdir(self.main_folder)
        for fold in range(self.n_split):
            os.mkdir(os.path.join(self.main_folder, f"fold_{fold + 1}"))
        self.write_model_structure()

    def vectorize_training_data(self, path_to_data, valuename, external_test_set=None,
                                save_features=False, featurize_threads=10):
        self.path_to_data = path_to_data
        self.valuename = valuename
        self.external_test_set = external_test_set
        self.save_features = save_features
        self.featurize_threads = featurize_threads
        self.prepare_out_folder()

        if self.path_to_data.endswith(".sdf"):
            self.object_type = "molecules"
            self.dataset = self.featurize_sdf(self.path_to_data)
        else:
            raise ValueError("Wrong input")

    def get_target_values(self, path_to_data, valuename):
        suppl = Chem.SDMolSupplier(path_to_data)
        mols = [x for x in suppl if x is not None]
        trg = [Chem.Mol.GetProp(mol, valuename) for mol in mols]

        if self.mode == "classification":
            classes = np.unique(trg)
            if len(classes) > 2:
                raise ValueError("multiclass not implemented")
            labels = {classes[i]: i for i in range(len(classes))}
            trg = [float(labels[i]) for i in trg]  # further training doesn't like int64
        elif self.mode == "regression":
            trg = np.array(trg, dtype=np.float)
        else:
            raise ValueError("mode must be classification or regression")

        return trg

    def featurize_sdf(self, path_to_data):
        suppl = Chem.SDMolSupplier(path_to_data)
        mols = [x for x in suppl if x is not None]

        smiles_strings = np.array([Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in mols])
        trg = self.get_target_values(path_to_data, self.valuename)

        tor = featurize_smiles_np(smiles_strings, ConvMolFeaturizer())
        x, y = shuffle(tor, trg, random_state=42)
        dataset = NumpyDataset(X=np.array(x), y=np.array(y))

        return dataset

    def split_dataset(self):

        if self.frac_train != 1:
            X_train, X_test, y_train, y_test = train_test_split(self.dataset.X, self.dataset.y,
                                                                test_size=1 - self.frac_train, random_state=7)
            self.test_set = NumpyDataset(X=X_test, y=y_test)
            self.train_set = NumpyDataset(X=X_train, y=y_train)
        else:
            self.train_set = self.dataset

        if self.mode == "regression":
            splitter = KFold(n_splits=self.n_split)
        else:
            splitter = StratifiedKFold(n_splits=self.n_split)

        for train_index, test_index in splitter.split(self.train_set.X, self.train_set.y):
            X_train, X_test = self.train_set.X[train_index], self.train_set.X[test_index]
            y_train, y_test = self.train_set.y[train_index], self.train_set.y[test_index]
            self.train_valid_split.append([NumpyDataset(X=X_train, y=y_train),
                                           NumpyDataset(X=X_test, y=y_test)])

    def calc_pos_weight(self):
        weights = compute_class_weight('balanced',
                                       np.unique(self.train_set.y),
                                       self.train_set.y)
        self.parameters["pos_weights"] = weights[-1]

    def find_best_fold(self):
        num_folds = len([i for i in os.listdir(self.restore_folder) if i.startswith("fold")])
        val_metric_values = np.zeros(num_folds)
        for fold in os.listdir(self.restore_folder):
            if fold.startswith("fold"):
                metrics_df = pd.read_csv(os.path.join(self.restore_folder, fold, "losses.csv"))
                val_metric_values[int(fold.split("_")[-1]) - 1] = np.max(metrics_df["valid_metric"])
        self.best_restore_fold = "fold_{}".format(np.argmax(val_metric_values) + 1)

    def train_cv_models(self, optimization=False):
        self.split_dataset()
        for fold, train_valid in enumerate(self.train_valid_split):
            backend.clear_session()
            params = self.parameters
            params["model_dir"] = os.path.join(self.main_folder, "fold_{}".format(fold + 1))
            print(params)
            model = self.model_class(**params)

            if self.restore_folder:
                model.restore(model_dir=os.path.join(self.restore_folder, self.best_restore_fold))

            model = self.train_model(model, train_valid[0], train_valid[1], self.n_epochs, params["model_dir"],
                                     optimization=optimization)
            self.models.append(model)
        return self.models

    def train_model(self, model, train_data, valid_data, n_epochs, out_folder, optimization=False):
        reduce_lr_wait = 10
        best_valid_loss = math.inf
        es_steps_cur = self.es_steps
        self.lr = self.parameters["learning_rate"]

        losses = {"train_loss": [],
                  "train_metric": [],
                  "valid_loss": [],
                  "valid_metric": [],
                  }
        if self.test_set:
            losses["test_loss"] = []
            losses["test_metric"] = []

        best_model = model

        if self.layers_to_freeze and self.restore_folder:
            variables_to_retrain = [i for i in tf.trainable_variables() if
                                    i.name.split("/")[0] not in self.layers_to_freeze]
        else:
            variables_to_retrain = None

        for i in range(n_epochs):
            reduce_lr_wait -= 1
            model.fit(train_data, nb_epoch=1, variables=variables_to_retrain, max_checkpoints_to_keep=None)
            if self.mode == "regression":
                train_loss = mean_squared_error(train_data.y, model.predict(train_data)) ** 0.5
                train_metric = r2_score(train_data.y, model.predict(train_data))
                valid_loss = mean_squared_error(valid_data.y, model.predict(valid_data)) ** 0.5
                valid_metric = r2_score(valid_data.y, model.predict(valid_data))
                if self.test_set:
                    test_loss = mean_squared_error(self.test_set.y, model.predict(self.test_set)) ** 0.5
                    test_metric = r2_score(self.test_set.y, model.predict(self.test_set))
            elif self.mode == "classification":
                train_loss = log_loss(train_data.y, np.argmax(model.predict(train_data), axis=-1))
                train_metric = matthews_corrcoef(train_data.y, np.argmax(model.predict(train_data), axis=-1))
                valid_loss = log_loss(valid_data.y, np.argmax(model.predict(valid_data), axis=-1))
                valid_metric = matthews_corrcoef(valid_data.y, np.argmax(model.predict(valid_data), axis=-1))
                if self.test_set:
                    test_loss = log_loss(self.test_set.y, np.argmax(model.predict(self.test_set), axis=-1).flatten())
                    test_metric = matthews_corrcoef(self.test_set.y,
                                                    np.argmax(model.predict(self.test_set), axis=-1).flatten())
            else:
                raise ValueError("mode must be classification or regression")

            losses["train_loss"].append(train_loss)
            losses["train_metric"].append(train_metric)
            losses["valid_loss"].append(valid_loss)
            losses["valid_metric"].append(valid_metric)
            if self.test_set:
                losses["test_loss"].append(test_loss)
                losses["test_metric"].append(test_metric)

            print("Epoch {}: training loss: {}, valid loss: {}".format(i, train_loss, valid_loss))

            if best_valid_loss > valid_loss:
                if not optimization:
                    model.save_checkpoint(model_dir="{}".format(out_folder),
                                          max_checkpoints_to_keep=1)
                best_valid_loss = valid_loss
                es_steps_cur = self.es_steps
                best_model = model
            else:
                es_steps_cur -= 1

                if es_steps_cur % int(self.es_steps / 4) == 0 and reduce_lr_wait < 0 and self.reduce_lr:
                    self.lr = self.lr / 2
                    model.set_lr(self.lr)
                    print("Reduce lr on plateu, current lr = {}".format(self.lr))

            if es_steps_cur == 0:
                print("Early stopping")
                break

        data_dict = {"train_true": train_data.y.tolist(),
                     "train_pred": best_model.predict(train_data).tolist(),
                     "valid_true": valid_data.y.tolist(),
                     "valid_pred": best_model.predict(valid_data).tolist(),
                     }
        if self.test_set:
            data_dict["test_true"] = self.test_set.y.tolist()
            data_dict["test_pred"] = best_model.predict(self.test_set).tolist()

        if not optimization:
            pd.DataFrame.from_dict(losses).to_csv("{}/losses.csv".format(out_folder))

            with open("{}/data.json".format(out_folder), 'w') as f:
                json.dump(data_dict, f)
        return best_model

    def make_cm(self, cm_data):
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.4)
        return sn.heatmap(cm_data, annot=True, annot_kws={"size": 16})

    def make_regplot(self, true_vals, pred_vals):
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.4)

        x, y = pd.Series(np.asarray(true_vals).flatten(), name="{}_true".format(self.valuename)), \
               pd.Series(np.asarray(pred_vals).flatten(), name="{}_pred".format(self.valuename))

        return sn.regplot(x=x, y=y)

    def make_plots_metrics(self, data):
        results = {}
        graphs = []

        if self.mode == "classification":
            for metric in CLASS_METRICS:
                results["{}_{}".format("train", metric)] = CLASS_METRICS[metric](data["train_true"],
                                                                                 np.argmax(data["train_pred"],
                                                                                           axis=-1))
                results["{}_{}".format("valid", metric)] = CLASS_METRICS[metric](data["valid_true"],
                                                                                 np.argmax(data["valid_pred"],
                                                                                           axis=-1))
                if self.test_set:
                    results["{}_{}".format("test", metric)] = CLASS_METRICS[metric](data["test_true"],
                                                                                    np.argmax(data["test_pred"],
                                                                                              axis=-1))
        elif self.mode == "regression":
            for metric in REG_METRICS:
                results["{}_{}".format("train", metric)] = REG_METRICS[metric](data["train_true"], data["train_pred"])
                results["{}_{}".format("valid", metric)] = REG_METRICS[metric](data["valid_true"], data["valid_pred"])
                if self.test_set:
                    results["{}_{}".format("test", metric)] = REG_METRICS[metric](data["test_true"], data["test_pred"])

        for key in results.keys():
            if "confusion" in key:
                results[key] = results[key].tolist()

        if self.mode == "classification":
            train_cm_fig = self.make_cm(pd.DataFrame(results["train_confusion_matrix"], range(2), range(2)))
            valid_cm_fig = self.make_cm(pd.DataFrame(results["valid_confusion_matrix"], range(2), range(2)))
            graphs.append(train_cm_fig)
            graphs.append(valid_cm_fig)
            if self.test_set:
                test_cm_fig = self.make_cm(pd.DataFrame(results["test_confusion_matrix"], range(2), range(2)))
                graphs.append(test_cm_fig)

        elif self.mode == "regression":
            true_regplot = self.make_regplot(data["train_true"], data["train_pred"])
            valid_regplot = self.make_regplot(data["valid_true"], data["valid_pred"])
            graphs.append(true_regplot)
            graphs.append(valid_regplot)
            if self.test_set:
                test_regplot = self.make_regplot(data["test_true"], data["test_pred"])
                graphs.append(test_regplot)

        return results, graphs

    def make_mean_plots_metrics(self):
        mean_res = {"train_true": [], "train_pred": [],
                    "valid_true": [], "valid_pred": [],
                    "test_true": [], "test_pred": []}

        for fold in range(self.n_split):
            with open(os.path.join(self.main_folder, "fold_{}".format(fold + 1), "data.json")) as jf:
                fold_res = json.load(jf)
            if self.mode == "regression":
                mean_res["train_pred"].append(np.asarray(self.models[fold].predict(self.train_set)))
                mean_res["valid_pred"].append(fold_res["valid_pred"])
                if self.test_set:
                    mean_res["test_pred"].append(fold_res["test_pred"])
                mean_res["valid_true"].append(fold_res["valid_true"])
            elif self.mode == "classification":
                # mean_res["train_pred"].append(np.argmax(self.models[fold].predict(self.train_set), axis=-1).flatten())
                mean_res["train_pred"].append(self.models[fold].predict(self.train_set))
                mean_res["valid_pred"].append(np.asarray(fold_res["valid_pred"]).flatten())
                if self.test_set:
                    mean_res["test_pred"].append(self.models[fold].predict(self.test_set))
                mean_res["valid_true"].append(fold_res["valid_true"])
        mean_res["train_true"] = self.train_set.y
        mean_res["valid_true"] = list(chain.from_iterable(mean_res["valid_true"]))
        mean_res["train_pred"] = np.mean(mean_res["train_pred"], axis=0)
        if self.test_set:
            mean_res["test_true"] = self.test_set.y
            mean_res["test_pred"] = np.mean(mean_res["test_pred"], axis=0)

        if self.mode == "classification":
            mean_res["valid_pred"] = list(chain.from_iterable(mean_res["valid_pred"]))
            mean_res["valid_pred"] = np.asarray(mean_res["valid_pred"]).reshape(-1, 2)
        elif self.mode == "regression":
            mean_res["valid_pred"] = list(chain.from_iterable(mean_res["valid_pred"]))

        mean_metrics, mean_graphs = self.make_plots_metrics(mean_res)
        return mean_metrics, mean_graphs

    def post_plots_metrics(self):
        # TODO make check for training finishing here

        for fold in range(self.n_split):
            output_path = os.path.join(self.main_folder, "fold_{}".format(fold + 1))

            with open(os.path.join(output_path, "data.json")) as jf:
                results, graphs = self.make_plots_metrics(json.load(jf))

            with open("{}/metrics.json".format(output_path), "w") as fp:
                json.dump(results, fp)

            for i in zip(graphs, ["Train", "Valid", "Test"]):
                i[0].figure.savefig(os.path.join(output_path, "{}_plot.png".format(i[1])))

        mean_metrics, mean_graphs = self.make_mean_plots_metrics()
        with open("{}/mean_metrics.json".format(self.main_folder), "w") as fp:
            json.dump(mean_metrics, fp)

        for i in zip(mean_graphs, ["Train", "Valid", "Test"]):
            i[0].figure.savefig(os.path.join(self.main_folder, "{}_mean_plot.png".format(i[1])))

    def post_loss_history(self):
        return
