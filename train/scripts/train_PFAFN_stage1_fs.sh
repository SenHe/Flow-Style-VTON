python -m torch.distributed.launch --nproc_per_node=1 --master_port=4703 train_PFAFN_stage1_fs.py --name PFAFN_stage1_fs  \
--PBAFN_warp_checkpoint 'checkpoints_fs/PBAFN_e2e_fs/PBAFN_warp_epoch_101.pth' --PBAFN_gen_checkpoint 'checkpoints_fs/PBAFN_e2e_fs/PBAFN_gen_epoch_101.pth'  \
--lr 0.00003 --niter 100 --niter_decay 100 --resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 1 --label_nc 14 --launcher pytorch --checkpoints_dir checkpoints_fs










