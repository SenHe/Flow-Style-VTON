python -m torch.distributed.launch --nproc_per_node=1 --master_port=7129 train_PBAFN_stage1_fs.py --name PBAFN_stage1_fs   \
--resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 1 --label_nc 14 --launcher pytorch --checkpoints_dir checkpoints_fs










