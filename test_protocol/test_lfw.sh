python test_lfw_srt.py \
    --test_set 'LFW' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '/CIS20/lyx/FaceX-Zoo-main-new/training_mode/backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:1' \
    --model_path "/CIS20/lyx/Self-restrained-Triplet-Loss-master/weights/weightsResNet50/SRT/weights.pt"
#    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir21/Epoch_19.pt"
#    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/addition_module/HSST/out_dir/Epoch_79.pt"
#    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/teacher_model/295672backbone.pth"