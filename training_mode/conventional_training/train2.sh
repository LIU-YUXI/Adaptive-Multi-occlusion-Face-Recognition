mkdir 'log2'
python train_webface_kd_adapt.py \
    --data_root '/CIS20/lyx/FaceX-Zoo-main-new/data/CASIA-WebFace' \
    --train_file '/CIS20/lyx/FaceX-Zoo-main-new/training_mode/webface_train_list.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'ArcFace' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir2' \
    --epoches 18 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 64 \
    --momentum 0.9 \
    --log_dir 'log2' \
    --tensorboardx_logdir 'mv-hrnet' \
    --device 'cuda:1' \
    2>&1 | tee log2/log.log