python test_lfw_kd.py \
    --test_set 'LFW' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:0' \
    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir2/Epoch_0.pt"