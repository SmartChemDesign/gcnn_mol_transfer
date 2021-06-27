import os
import tensorflow as tf
from keras import backend
import sys

sys.path.append("./Source")
from Keras_graphconv import GraphConvModelMod
from TransferTrainer import TransferTrainer

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

backend.clear_session()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

HYPERPARAMETERS = {"classification": {"n_tasks": 1, "graph_conv_layers": [128, 64], "dense_layer_size": 64,
                                      "dropout": 0, "mode": "classification", "number_atom_features": 75,
                                      "n_classes": 2, "batch_size": 64, "num_dense": 3, "learning_rate": 0.001,
                                      "transfer": False},
                   "regression": {"n_tasks": 1, "graph_conv_layers": [128, 64], "dense_layer_size": 64,
                                  "dropout": 0.0, "mode": "regression", "number_atom_features": 75,
                                  "uncertainty": False, "learning_rate": 0.001,
                                  "batch_size": 10, "num_dense": 3, "transfer": False}}

path_to_data = "logS_logP_dataset.sdf"
valuename = "logS"
source_fold_folder = ""
output_folder = "./Output"
mode = "regression"

trainer = TransferTrainer(GraphConvModelMod, mode=mode,
                          output_folder=output_folder, n_epochs=1000, n_split=5,
                          es_steps=20, parameters=HYPERPARAMETERS[mode],
                          frac_train=0.8, batch_size=100,
                          )

# trainer.restore_model_params(source_fold_folder, layers_to_freeze=["graph_conv"])
trainer.vectorize_training_data(path_to_data, valuename)
trainer.train_cv_models()
trainer.post_plots_metrics()
