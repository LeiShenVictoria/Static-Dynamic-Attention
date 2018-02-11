# Static-Dynamic-Attention

## Prerequisites
* Python 3.6
* Pytorch 0.3.0
* CUDA 8.0

## Getting Started

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

