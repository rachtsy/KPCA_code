## Code for RPC-SymViT and Scaled Attention in Our Paper
Unveiling the Hidden Structure of Self-Attention via Kernel Principal Component Analysis

https://arxiv.org/abs/2406.13762

## Requirements
This toolkit requires PyTorch `torch`. 

The experiments for the paper were conducted with Python 3.10.12, timm 0.9.12 and PyTorch >= 1.4.0.

The toolkit supports [Weights & Biases](https://docs.wandb.ai/) for monitoring jobs. If you use it, also install `wandb`. 

## Instructions
Please run each command line in the respective folders. A run.sh script is provided there as well. 

The hyper parameters that may be tuned are 
1. --num_iter: the number of iterations of the PAP algorithm to run in a RPA-Attention layer
2. --lambd: the regularization parameter that controls the sparsity of the corruption matrix S
3. --layer: the layer to implement RPA-Attention, choose -1 for all layers

### Training

RPC-SymViT
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env main_train.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/data/imagenet --output_dir /path/to/checkpoints/ --robust --num_iter 4 --lambd 4 --layer 0
```

Scaled Attention *S*
```
CUDA_VISIBLE_DEVICES='1,2,3,0' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env main_train.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet/ --output_dir /path/to/output/directory/
```

Scaled Attention $\alpha$ $\times$ Asym \
Running this script without --s_scalar will default to training Scaled Attention *S*
```
CUDA_VISIBLE_DEVICES='1,2,3,0' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env main_train.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet/ --output_dir /path/to/output/directory/ --s_scalar
```

### Robustness Evaluation 
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env eval_OOD.py --model deit_tiny_patch16_224 --data-path /path/to/data/imagenet/ --output_dir /path/to/checkpoints/ --robust --num_iter 4 --lambd 4 --layer 0 --resume /path/to/model/checkpoint/
```

### Attack Evaluation
Run with --attack 'fgm' for FGSM attack and adjust --eps for severity of perturbation. 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port_num --use_env main.py --model deit_tiny_patch16_224 --batch-size 48 --data-path /path/to/data/imagenet --output_dir /path/to/output/directory/ --project_name 'project_name' --job_name job_name --attack 'pgd' --eps 0.1 --finetune /path/to/trained/model/ --eval 1 --robust --num_iter 2 --layer -1 --lambd 4
```

### Reconstruction Error Code (Sec 2.2.1)
Run script run.py in the Reconstruction folder with wandb to reproduce the plot in Sec 2.2.1 of the paper. \
The loss is calculated for the same batch of images for each epoch. \
In the first 5-10 epochs, the loss logged may increase but should decrease afterwards. 

```
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env main_train.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet/ --output_dir /path/to/output/directory/ --lr 1e-4 --warmup-epochs 0 
```