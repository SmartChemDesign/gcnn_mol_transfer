import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, BatchNormalization
from tensorflow.nn import weighted_cross_entropy_with_logits
import logging
from deepchem.models.layers import GraphConv, GraphPool, GraphGather, SwitchedDropout
from deepchem.feat.mol_graphs import ConvMol
import deepchem as dc
from deepchem.metrics import to_one_hot
import numpy as np
from deepchem.models.graph_models import TrimGraphOutput, GraphConvModel, KerasModel
import os
from tensorflow.python.framework import ops
import collections
from rdkit import Chem
from deepchem.data.data_loader import featurize_smiles_np
from deepchem.feat import ConvMolFeaturizer, AdjacencyFingerprint
from deepchem.data import NumpyDataset, DiskDataset
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, BinaryCrossEntropy, Loss, _make_shapes_consistent
import random
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from deepchem.models.optimizers import Adam
import sys

logger = logging.getLogger(__name__)
try:
    from collections.abc import Sequence
except:
    from collections import Sequence

sys.path.append("./")


ops.reset_default_graph()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OSDR_LOG_FOLDER"] = "logs"


class GraphConvModelMod(KerasModel):

    def __init__(self,
                 n_tasks,
                 graph_conv_layers=None,
                 dense_layer_size=128,
                 dropout=0.0,
                 mode="classification",
                 number_atom_features=75,
                 n_classes=2,
                 uncertainty=False,
                 batch_size=10,
                 model_dir=None,
                 pos_weight=None,
                 num_dense=2,
                 transfer=False,
                 **kwargs):
        """
    This is modified version of deepchem GraphConvModel class

    Parameters
    ----------
    n_tasks: int
      Number of tasks
    graph_conv_layers: list of int
      Width of channels for the Graph Convolution Layers
    dense_layer_size: int
      Width of channels for Atom Level Dense Layer before GraphPool
    dropout: list or float
      the dropout probablity to use for each layer.  The length of this list should equal
      len(graph_conv_layers)+1 (one value for each convolution layer, and one for the
      dense layer).  Alternatively this may be a single value instead of a list, in which
      case the same value is used for every layer.
    mode: str
      Either "classification" or "regression"
    number_atom_features: int
        75 is the default number of atom features created, but
        this can vary if various options are passed to the
        function atom_features in graph_features
    n_classes: int
      the number of classes to predict (only used in classification mode)
    uncertainty: bool
      if True, include extra outputs and loss terms to enable the uncertainty
      in outputs to be predicted
    """
        if graph_conv_layers is None:
            graph_conv_layers = [64, 64]

        if mode not in ['classification', 'regression']:
            raise ValueError("mode must be either 'classification' or 'regression'")
        self.n_tasks = n_tasks
        self.mode = mode
        self.num_dense = num_dense
        self.dense_layer_size = dense_layer_size
        self.graph_conv_layers = graph_conv_layers
        self.number_atom_features = number_atom_features
        self.n_classes = n_classes
        self.uncertainty = uncertainty
        self.transfer = transfer
        self.pos_weight = None
        if not isinstance(dropout, collections.Sequence):
            dropout = [dropout] * (len(graph_conv_layers) + 1)
        if len(dropout) != len(graph_conv_layers) + 1:
            raise ValueError('Wrong number of dropout probabilities provided')
        self.dropout = dropout
        if uncertainty:
            if mode != "regression":
                raise ValueError("Uncertainty is only supported in regression mode")
            if any(d == 0.0 for d in dropout):
                raise ValueError(
                    'Dropout must be included in every layer to predict uncertainty')

        # Build the model.

        # if self.transfer:
        #     mulipliers = self.make_lr_multipliers()
        #     print(mulipliers)
        #     self.optimizer = LearningRateMultiplier(Adam, lr_multipliers=mulipliers)

        atom_features = Input(shape=(self.number_atom_features,))
        degree_slice = Input(shape=(2,), dtype=tf.int32)
        membership = Input(shape=tuple(), dtype=tf.int32)
        n_samples = Input(shape=tuple(), dtype=tf.int32)
        dropout_switch = tf.keras.Input(shape=tuple())
        self.deg_adjs = []
        for i in range(0, 10 + 1):
            deg_adj = Input(shape=(i + 1,), dtype=tf.int32)
            self.deg_adjs.append(deg_adj)
        in_layer = atom_features

        for layer_size, dropout in zip(self.graph_conv_layers, self.dropout):
            gc1_in = [in_layer, degree_slice, membership] + self.deg_adjs
            gc1 = GraphConv(layer_size, activation_fn=tf.nn.relu, )(gc1_in)
            batch_norm1 = BatchNormalization(fused=False)(gc1)
            if dropout > 0.0:
                batch_norm1 = SwitchedDropout(rate=self.dropout[-1])(  # TODO hardcoded dropout here
                    [batch_norm1, dropout_switch])
            gp_in = [batch_norm1, degree_slice, membership] + self.deg_adjs
            in_layer = GraphPool()(gp_in)

        for i in range(int(self.num_dense)):
            in_layer = Dense(self.dense_layer_size, activation=tf.nn.relu)(in_layer)
            in_layer = BatchNormalization(fused=False, )(in_layer)
            if self.dropout[-1] > 0.0:
                in_layer = SwitchedDropout(rate=self.dropout[-1])(
                    [in_layer, dropout_switch])
        self.neural_fingerprint = GraphGather(
            batch_size=batch_size,
            activation_fn=tf.nn.tanh, )([in_layer, degree_slice, membership] +
                                        self.deg_adjs)

        n_tasks = self.n_tasks
        if self.mode == 'classification':
            n_classes = self.n_classes
            logits = Reshape((n_tasks, n_classes))(Dense(n_tasks * n_classes)(
                self.neural_fingerprint))
            logits = TrimGraphOutput()([logits, n_samples])
            output = Softmax()(logits)
            outputs = [output, logits]
            output_types = ['prediction', 'loss']
            if self.pos_weight:
                loss = WeightedSigmoidCrossEntropy(self.pos_weight)
            else:
                # loss = MacroSoftF1()
                loss = SoftmaxCrossEntropy()
        else:
            output = Dense(n_tasks, trainable=False)(self.neural_fingerprint)
            output = TrimGraphOutput()([output, n_samples])
            if self.uncertainty:
                log_var = Dense(n_tasks)(self.neural_fingerprint)
                log_var = TrimGraphOutput()([log_var, n_samples])
                var = Activation(tf.exp)(log_var)
                outputs = [output, var, output, log_var]
                output_types = ['prediction', 'variance', 'loss', 'loss']

                def loss(outputs, labels, weights):
                    diff = labels[0] - outputs[0]
                    return tf.reduce_mean(diff * diff / tf.exp(outputs[1]) + outputs[1])
            else:
                outputs = [output]
                output_types = ['prediction']
                loss = L2Loss()

        self.keras_model = tf.keras.Model(
            inputs=[atom_features, degree_slice, membership, n_samples, dropout_switch
                    ] + self.deg_adjs,
            outputs=outputs)
        super(GraphConvModelMod, self).__init__(
            self.keras_model, loss, output_types=output_types, batch_size=batch_size, **kwargs)

    def set_lr(self, new_lr):
        self.optimizer = Adam(learning_rate=new_lr)

    def make_lr_multipliers(self):
        multipliers_dict = {}
        for conv in enumerate(self.graph_conv_layers[1:]):  # TODO hardcoded number of freezed conv layers here
            multipliers_dict[f"graph_conv_{conv[0] + 1}"] = 0

        multipliers_dict[f"dense"] = 0

        for dense in range(self.num_dense - 1):
            multipliers_dict[f"dense_{dense + 1}"] = 0
        return multipliers_dict

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        for epoch in range(epochs):
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
                    batch_size=self.batch_size,
                    deterministic=deterministic,
                    pad_batches=pad_batches):
                if self.mode == 'classification':
                    y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                        -1, self.n_tasks, self.n_classes)
                multiConvMol = ConvMol.agglomerate_mols(X_b)
                n_samples = np.array(X_b.shape[0])
                if mode == 'predict':
                    dropout = np.array(0.0)
                else:
                    dropout = np.array(1.0)
                inputs = [
                    multiConvMol.get_atom_features(), multiConvMol.deg_slice,
                    np.array(multiConvMol.membership), n_samples, dropout
                ]
                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
                yield (inputs, [y_b], [w_b])


def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """

    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels

    return macro_cost


class MacroSoftF1(Loss):
    """

    """
    def __call__(self, output, labels):
        output, labels = _make_shapes_consistent(output, labels)
        return macro_soft_f1(labels, output)


class WeightedSigmoidCrossEntropy(Loss):
    """The cross entropy between pairs of probabilities.

    The arguments should each have shape (batch_size) or (batch_size, tasks).  The
    labels should be probabilities, while the outputs should be logits that are
    converted to probabilities using a sigmoid function.
    """

    def __init__(self, pos_weight):
        self.pos_weight = pos_weight

    def __call__(self, output, labels):
        output, labels = _make_shapes_consistent(output, labels)
        return weighted_cross_entropy_with_logits(
            labels, output, reduction=tf.losses.Reduction.NONE, pos_weight=self.pos_weight)


def get_datasets(path_to_sdf, valuename, n_splits=5, verbose=False, frac_train=0.8, subsample_size=None, save=None,
                 load=None, full=False, classification=False):
    """

    @param path_to_sdf:
    @param valuename:
    @param n_splits:
    @param verbose:
    @param frac_train:
    @param subsample_size:
    @param save:
    @param load:
    @param full:
    @param classification:
    @return:
    """
    if load:
        db = NumpyDataset.from_json(load)
    else:
        suppl = Chem.SDMolSupplier(path_to_sdf)
        mols = [x for x in suppl if x is not None]
        if subsample_size:
            mols = random.sample(mols, subsample_size)

        smiles_strings = np.array([Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in mols])
        trg = [Chem.Mol.GetProp(mol, valuename) for mol in mols]

        if len(np.unique(trg)) == 2:
            labels = {np.unique(trg)[i]: i for i in range(len(np.unique(trg)))}
            trg = [float(labels[i]) for i in trg]
            classification = True
        else:
            trg = np.array([float(Chem.Mol.GetProp(mol, valuename)) for mol in mols])

        tor = featurize_smiles_np(smiles_strings, ConvMolFeaturizer())

        db = NumpyDataset(X=np.array(tor), y=np.array(trg))
        if save:
            NumpyDataset.to_json(db, save)
    if full:
        return db

    if not classification:
        splitter = dc.splits.splitters.RandomSplitter(verbose=verbose)
        train, test = splitter.train_test_split(db, frac_train=frac_train, seed=42)
        train_valid_sets = splitter.k_fold_split(train, n_splits)
    else:
        train_valid_sets = []
        X_train, X_test, y_train, y_test = train_test_split(db.X, db.y, test_size=1 - frac_train, random_state=42)
        test = NumpyDataset(X=X_test, y=y_test)
        train = NumpyDataset(X=X_train, y=y_train)

        splitter = StratifiedKFold(n_splits=n_splits)
        for train_index, test_index in splitter.split(train.X, train.y):
            X_train, X_test = train.X[train_index], train.X[test_index]
            y_train, y_test = train.y[train_index], train.y[test_index]
            train_valid_sets.append([NumpyDataset(X=X_train, y=y_train),
                                     NumpyDataset(X=X_test, y=y_test)])

    return train, test, train_valid_sets
