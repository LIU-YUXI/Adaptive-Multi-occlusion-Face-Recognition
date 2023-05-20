mkdir 'log'
python train.py \
    --data_root '/CIS20/lyx/FaceX-Zoo-main-new/data/CASIA-WebFace' \
    --data_mask_root '/CIS20/lyx/FaceX-Zoo-main-new/data/CASIA-WebFace_random_mask' \
    --train_file '/CIS20/lyx/FaceX-Zoo-main-new/training_mode/webface_train_list.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '/CIS20/lyx/FaceX-Zoo-main-new/training_mode/backbone_conf.yaml' \
    --head_type 'ArcFace' \
    --head_conf_file '/CIS20/lyx/FaceX-Zoo-main-new/training_mode/head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir' \
    --epoches 80 \
    --step '30, 50, 70' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 64 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'mv-hrnet' \
    --device 'cuda:1' \
    2>&1 | tee log/log.log
#    --resume \
#    --pretrain_model "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir19/Epoch_0.pt" \
#    --pretrain_adapt "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir19/Epoch_0_adapt.pt" \
#    --pretrain_header "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir19/Epoch_0_header.pt" \