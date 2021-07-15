CUDA_VISIBLE_DEVICES=0 python run.py \
 --do_train --use_dm --use_pm --use_rt --use_name --use_bio --use_network --use_count \
 --train_data_file=data/train-data-camera-ready.json \
 --val_data_file=data/train-data-camera-ready.json \
 --output_dir=data \
 --checkpoint_interval=50000 --valid_interval=50000 --batch_size=2 \
 --classifier_config_dir=data/bert-config.json

CUDA_VISIBLE_DEVICES=0 python run.py \
 --do_infer --use_dm --use_pm --use_rt --use_name --use_bio --use_network --use_count \
 --infer_data_file=data/sample_outputs.json \
 --output_dir=data \
 --model_dir=data/full-model.pth \
 --checkpoint_interval=50000 --valid_interval=50000 --batch_size=2 \
 --classifier_config_dir=data/bert-config.json
