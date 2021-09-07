python -m torch.distributed.launch --nproc_per_node=1 --master_port=4736 train_PBAFN_e2e_fs.py --name PBAFN_e2e_fs   \
--PBAFN_warp_checkpoint 'checkpoints_fs/PBAFN_stage1_fs/PBAFN_warp_epoch_101.pth' --resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 1 --label_nc 14 --launcher pytorch --checkpoints_dir checkpoints_fs










