python test_lfw_case.py \
    --test_set 'CASE' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 6 \
    --device 'cuda:0' \
    --adapt_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir30/Epoch_19_adapt.pt" \
    --model_path "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir30/Epoch_19.pt"