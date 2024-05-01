# Diabetic Retinopathy Detection: CNNs vs. Transformers
This codebase is the official implementation of Diabetic Retinopathy Detection using Cross-Domain Colored Fundus Photography: CNNs vs. Transformers.
- Explore the generalizability of CNNs and Transformers.
- Apply test-time adaptation (TTA) to enhance the performance in target domains.
## Installation
- CUDA/Python
## Quick Start
(1) Download the dataset.

(2) Train different models on the source domain.
```python
python train.py --root "data_path" --dataset "DR2015" --output ".\results" --network "resnet34" --base_lr 1e-3 --batchsize 32 
```
Note: change `--output ".\results"` to change your path to save your model.  
Note: change `--network "resnet34"` for training on different models (e.g. `resnet-34, resnet-50, efficientnet-b1, efficientnet-b5, vit-t16, vit-s16, swinv2, convit-s`).  
Note: change `--base_lr 1e-3` to adjust the initial learning rate. 

(3) Test different models on the target domain.
```python
python test.py --root "data_path" --dataset "DR2015" --network "resnet34"  --algorithm "TTFA" --use_cuda True
```

(4) Visualization.
```python
python visualization.py --image_path "image_path" --network "resnet34" --use_cuda True
```
