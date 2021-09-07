python -m torch.distributed.launch --nproc_per_node=1 --master_port=7129 train_PFAFN_e2e_fs.py --name PFAFN_e2e_fs   \
--PFAFN_warp_checkpoint 'checkpoints_fs/PFAFN_stage1_fs/PFAFN_warp_epoch_201.pth'  \
--PBAFN_warp_checkpoint 'checkpoints_fs/PBAFN_e2e_fs/PBAFN_warp_epoch_101.pth' --PBAFN_gen_checkpoint 'checkpoints_fs/PBAFN_e2e_fs/PBAFN_gen_epoch_101.pth'  \
--resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 1 --label_nc 14 --launcher pytorch --checkpoints_dir checkpoints_fs










