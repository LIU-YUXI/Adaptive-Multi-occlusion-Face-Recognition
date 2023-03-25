python test_lfw_wo_adapt.py \
    --test_set 'MEGLASS' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:1' \
    --adapt_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir5/Epoch_15_adapt.pt" \
    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/teacher_model/295672backbone.pth"