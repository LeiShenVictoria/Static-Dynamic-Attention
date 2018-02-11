# Static-Dynamic-Attention

## Prerequisites
* Python 3.6
* Pytorch 0.3.0
* CUDA 8.0

## Getting Started

The ```data``` folder contains only training sets, test sets, and word2vec files for the example. <br>
You can use your own dataset, but it must be consistent with the data format in the sample file, if you do not want to make changes to the code.
### train static model
```bash
python3 train.py --type_model 1
```
### train dynamic model
```bash
python3 train.py --type_model 0
```
### predict 
```bash
python3 predict.py --weights static_parameters_IterEnd
```
#### or 
```bash
python3 predict.py --weights dyna_parameters_IterEnd
```

