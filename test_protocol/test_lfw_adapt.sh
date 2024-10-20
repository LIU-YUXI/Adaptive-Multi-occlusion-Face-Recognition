python test_lfw_adapt.py \
    --test_set 'CALFW_SUNGLASSES' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:2' \
    --adapt_path "/mnt/diskB/lyx/AMOFR/training_mode/conventional_training/out_dir15-ft/Epoch_16_adapt.pt" \
    --model_path "/mnt/diskB/lyx/AMOFR/training_mode/conventional_training/out_dir15-ft/Epoch_16.pt"

python test_lfw_adapt.py \
    --test_set 'CPLFW' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:2' \
    --adapt_path "/mnt/diskB/lyx/AMOFR/training_mode/conventional_training/out_dir15-ft/Epoch_16_adapt.pt" \
    --model_path "/mnt/diskB/lyx/AMOFR/training_mode/conventional_training/out_dir15-ft/Epoch_16.pt"

python test_lfw_adapt.py \
    --test_set 'LFW' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:2' \
    --adapt_path "/mnt/diskB/lyx/AMOFR/training_mode/conventional_training/out_dir15-ft/Epoch_16_adapt.pt" \
    --model_path "/mnt/diskB/lyx/AMOFR/training_mode/conventional_training/out_dir15-ft/Epoch_16.pt"

python test_lfw_adapt.py \
    --test_set 'MEGLASS' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:2' \
    --adapt_path "/mnt/diskB/lyx/AMOFR/training_mode/conventional_training/out_dir15-ft/Epoch_16_adapt.pt" \
    --model_path "/mnt/diskB/lyx/AMOFR/training_mode/conventional_training/out_dir15-ft/Epoch_16.pt"