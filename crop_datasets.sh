#!/bin/bash
#python3 src/crop_datasets.py model_type=vit_base dim=90 data_dir=/mnt/hdd/leon/datasets/ output_root=/mnt/hdd/leon/outputs/ dataset_name=cocostuff27
#python3 src/crop_datasets.py model_type=vit_small dim=70 data_dir=/mnt/data/datasets/ output_root=/mnt/data/outputs dataset_name=cocostuff27 depth_type=kbr
#python3 src/crop_datasets.py model_type=vit_small dim=70 data_dir=/mnt/hdd/leon/datasets/nyuv2/ output_root=/mnt/hdd/leon/outputs dataset_name=nyuv2 depth_type=zoedepth
#python3 src/crop_datasets.py model_type=vit_base dim=90 data_dir=/mnt/data/datasets/ output_root=/mnt/data/outputs dataset_name=nyuv2

python3 src/crop_datasets.py model_type=vit_small dim=70 data_dir=/mnt/data/datasets output_root=/mnt/data/datasets dataset_name=pascalvoc depth_type=zoedepth
