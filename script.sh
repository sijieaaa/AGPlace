

nohup python train.py --cuda 0  --dataset nuscenes  --camnames fl_f_fr_bl_b_br --otherloss_weight $otherloss_weight > nuscenes${otherloss_weight}.log 2>&1 &


nohup python train.py --cuda 1  --dataset kitti360  --camnames 00 --otherloss_weight $otherloss_weight > kitti360${otherloss_weight}.log 2>&1 &

