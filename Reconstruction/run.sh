CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 1 --nproc_per_node=4 --use_env main_train.py \
--model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet/ --output_dir /path/to/output/directory/ \
--lr 1e-4 --warmup-epochs 0

