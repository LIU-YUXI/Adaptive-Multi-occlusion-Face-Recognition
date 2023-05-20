mkdir 'log41'
python train_webface_kd_adapt.py \
    --data_root '/CIS20/lyx/FaceX-Zoo-main-new/data/CASIA-WebFace' \
    --train_file '/CIS20/lyx/FaceX-Zoo-main-new/training_mode/webface_train_list.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'ArcFace' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir41' \
    --epoches 20 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 48 \
    --momentum 0.9 \
    --log_dir 'log41' \
    --tensorboardx_logdir 'mv-hrnet' \
    --device 'cuda:0' \
    --w 100 \
    2>&1 | tee log41/log_sunglass_glass_skd1000_onlypred2.log
#    --resume \
#    --pretrain_model "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir17/Epoch_4.pt" \
#    --pretrain_adapt "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir17/Epoch_4_adapt.pt" \
#    --pretrain_header "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir17/Epoch_4_header.pt" \