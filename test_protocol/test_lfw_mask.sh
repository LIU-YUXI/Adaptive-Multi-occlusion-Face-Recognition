python test_lfw_kd.py \
    --test_set 'MEGLASS' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:1' \
    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir_kd_sunglasses/Epoch_19.pt"
#    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir21/Epoch_19.pt"
#    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/addition_module/HSST/out_dir/Epoch_79.pt"
#    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/teacher_model/295672backbone.pth"