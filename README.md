# gcnn_transfer_learning

This package provides tools for training graph convolutional models for molecules using 
transfer learning techniques.

[![alt text](https://pubs.acs.org/na101/home/literatum/publisher/achs/journals/content/jpclcd/2021/jpclcd.2021.12.issue-38/acs.jpclett.1c02477/20210923/images/medium/jz1c02477_0007.gif)](https://doi.org/10.1021/acs.jpclett.1c02477)
## Installation

Use [conda](https://conda.io/projects/conda/en/latest/index.html) to install requirements presented in **nb_deepchem_gpu.yml**
```bash
conda env create -f nb_deepchem_gpu.yml
```
## Usage
Training process carried out with **run_train.py** through TransferTrainer object. All training features are
controlled by several flags in trainer object. Transfer training carried out by adding path to folder of 
pretrained model. Input data samples are presented in **Datasets**

Flags for transfer training
```python
restore_folder=source_fold_folder, layers_to_freeze=["graph_conv"]
```

Flags for hyperopt optimization:

    hyperopt=True, hyperopt_evals=100

There are several mandatory input for TransferTrainer class init

    path_to_sdf - path to sdf file for molecules or to folder with cif for materials
    valuename - name of target property in sdf file, optional for materials
    source_fold_folder - path to pretrained model to use it as donor for transfer, only models trained with 
    TransferTrainer are allowed
    output_folder - path to folder where TransferTrainer will create dir with all training results
    mode - classification or regression based on target property, multiclass not implemented

## Citation
If you use this code in your research, please cite this paper [Size Doesn’t Matter: Predicting Physico- or Biochemical Properties Based on Dozens of Molecules](https://doi.org/10.1021/acs.jpclett.1c02477)

## License
[MIT](https://choosealicense.com/licenses/mit/)