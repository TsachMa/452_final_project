# Elem-XtalFormer (EXF)
https://docs.google.com/document/d/1Ik2R_p8RlXFN_Vl2qdooxouiQTnoraYKnIgERmSSRbU/edit

This implements the Elem-XtalFormer (EXF) model to predict crystal symmetry space groups from X-ray diffraction and chemical composition data

## Installation

- Clone the repository
- Use `conda` to install the environment with:
```
conda env create -f env.yml
```
- Activate the environment with:
```
conda activate exf
```

## Dataset

We use the dataset curated by (Xie. et al)[https://arxiv.org/abs/2110.06197]

Run the `data_prep.ipynb` notebook after downloading the data to create the required supplementary files 

## Model

### Training Elem2Vec

The `elem2vec.ipynb` notebook contains the necessary code to train and save the element composition embeddings necessary for the main EXF model

### Training EXF

The `exf.ipynb` notebook contains a simple example walkthrough on training EXF from scratch

### Experiments

