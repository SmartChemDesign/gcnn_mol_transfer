# gcnn_mol_transfer

This package provides tools for training graph convolutional models for molecules using 
transfer learning techniques. 

All necessary packages can be install via [Anaconda](https://anaconda.org/)

List of packages presented in **requirements.yml**

Training process carried out with **run_train.py** through TransferTrainer object. All training features are
controlled by several flags in trainer object. Transfer training carried out by adding path to folder of 
pretrained model. 


There are several mandatory input for TransferTrainer class init

    path_to_sdf - path to sdf file for molecules or to folder with cif for materials
    valuename - name of target property in sdf file, optional for materials
    source_fold_folder - path to pretrained model to use it as donor for transfer, only models trained with 
    TransferTrainer are allowed
    output_folder - path to folder where TransferTrainer will create dir with all training results
    mode - classification or regression based on target property, multiclass not implemented

Sample data samples are presented in **Datasets** and donor model in **Donor_models**

Additional model hyperparameters can be modified via HYPERPARAMETERS dictionary in **run_train.py**.

For transfer learning one should uncomment line
> trainer.restore_model_params(source_fold_folder, layers_to_freeze=["graph_conv"])

If you use this code, cite this paper https://doi.org/10.1021/acs.jpclett.1c02477



