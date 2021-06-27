from rdkit import Chem
import os
import numpy as np
import pandas as pd
import json
from deepchem.data.data_loader import featurize_smiles_np
from deepchem.feat import ConvMolFeaturizer
from deepchem.data import NumpyDataset

import sys
sys.path.append("./Keras_graphconv.py")

from Keras_graphconv import GraphConvModelMod


class Predictor:
    def __init__(self, model_folder, model_class=GraphConvModelMod, input_mols=None, valuename=None, write_sdf=False,
                 mode="classification"):
        self.model_class = model_class
        self.restore_params = {}
        self.model_folder = model_folder
        self.folds = []
        self.input_mols = input_mols
        # self.mode = self.model_folder.split("/")[-1].split("_")[0]
        self.mode = mode
        self.mols_num = None
        self.mols = []
        self.valuename = None
        self.featurized_data = None
        self.predicted_data = pd.DataFrame()

        if input_mols:
           self.add_dataset(input_mols, valuename)

        self.restore_models()

    def identify_mols(self, input_mols):
        if isinstance(input_mols, str):
            mols = Chem.SDMolSupplier(input_mols)
            self.mols_num = len(mols)
        elif isinstance(input_mols, list):
            mols = input_mols
            self.mols_num = len(mols)
        else:
            mols = [input_mols]
            self.mols_num = 1
        return mols

    def filter_bad_folds(self):
        # TODO add this later maybe
        return

    def get_restore_parameters(self):
        self.restore_params["n_tasks"] = 1 # TODO fix n_tasks issue
        self.restore_params["mode"] = self.mode
        with open(os.path.join(self.model_folder, "net_structure_data.json")) as jf:
            jd = json.load(jf)
        for key in jd:
            self.restore_params[key] = jd[key]

    def restore_models(self):
        self.get_restore_parameters()
        model_folds = [fold for fold in os.listdir(self.model_folder) if fold.count("fold")]

        for fold in model_folds:
            model = self.model_class(**self.restore_params)
            model.restore(model_dir=os.path.join(self.model_folder, fold))
            self.folds.append(model)

    def featurize_dataset(self):
        smiles_strings = np.array([Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in self.mols])
        tor = featurize_smiles_np(smiles_strings, ConvMolFeaturizer())
        dataset = NumpyDataset(X=np.array(tor))

        if self.valuename:
            trg = [Chem.Mol.GetProp(mol, self.valuename) for mol in self.mols]

            if self.mode == "classification":
                classes = np.unique(trg)
                if len(classes) > 2:
                    raise ValueError("multiclass not implemented")
                labels = {classes[i]: i for i in range(len(classes))}
                trg = np.asarray([float(labels[i]) for i in trg])  # further training doesn't like int64
            elif self.mode == "regression":
                trg = np.array([float(Chem.Mol.GetProp(mol, self.valuename)) for mol in self.mols])
            else:
                raise ValueError("mode must be classification or regression")

            dataset._y = trg

        return dataset

    def add_dataset(self, input_mols, valuename=None):
        self.valuename = valuename
        self.mols = self.identify_mols(input_mols)
        self.featurized_data = self.featurize_dataset()

    def predict(self):
        for fold in enumerate(self.folds):
            self.predicted_data["fold_{}".format(fold[0] + 1)] = np.asarray([i[0].tolist().index(max(i[0])) for i in fold[1].predict(self.featurized_data)])
        self.predicted_data['mean'] = self.predicted_data.mode(axis=1)
