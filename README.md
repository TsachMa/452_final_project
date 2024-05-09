# Elem-XtalFormer (EXF)

This implements the Elem-XtalFormer (EXF) model to predict crystal symmetry space groups from X-ray diffraction and chemical composition data

## Installation

- Clone the repository
- Use `conda` to install the environment by running:
```
conda env create -f env.yml
```
- Activate the environment by running:
```
conda activate exf
```

## Dataset

We use the dataset curated by [Xie. et al](https://arxiv.org/abs/2110.06197_)

Run the `data_prep.ipynb` notebook after downloading the relevant crystal structure data to create the required supplementary files 

## Model

### Training Elem2Vec

The `elem2vec.ipynb` notebook contains the necessary code to train and save the element composition embeddings necessary for the main EXF model

### Training EXF

The `exf.ipynb` notebook contains a simple example walkthrough on training EXF from scratch

`data_utils.py`: contains the PyTorch dataset class to load data for training and X-ray diffraction classes for experimental simulation of data

`transformer_models`: contains EXF model architechture classes

### Experiments


`regularization.py`: contains experiments related to L1/L2 regularization

`eval.ipynb` & `wo_comp_eval.ipynb`: contain experiments related to noise levels and the composition module


