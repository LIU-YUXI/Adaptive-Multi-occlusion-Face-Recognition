mkdir 'log_kd_sunglasses'
python train_webface_kd.py \
    --data_root '/CIS20/lyx/FaceX-Zoo-main-new/data/CASIA-WebFace' \
    --train_file '/CIS20/lyx/FaceX-Zoo-main-new/training_mode/webface_train_list.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'ArcFace' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir_kd_sunglasses' \
    --epoches 20 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 48 \
    --momentum 0.9 \
    --log_dir 'log_kd_sunglasses' \
    --tensorboardx_logdir 'mv-hrnet' \
    --device 'cuda:1' \
    2>&1 | tee log_kd_sunglasses/log3.log