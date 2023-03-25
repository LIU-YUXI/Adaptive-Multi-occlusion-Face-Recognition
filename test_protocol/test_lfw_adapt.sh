python test_lfw_adapt.py \
    --test_set 'LFW' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:1' \
    --adapt_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir5/Epoch_17_adapt.pt" \
    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir5/Epoch_17.pt"