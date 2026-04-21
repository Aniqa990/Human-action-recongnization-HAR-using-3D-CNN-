python util_scripts/generate_video_jpgs.py

python prepare_annotation.py

mkdir "G:\My Drive\SEG_DATASET_jpg\results"

python main.py --root_path "G:/My Drive/DATASET_jpg" --video_path . --annotation_path "G:/My Drive/DATASET_jpg/dataset.json" --result_path "G:/My Drive/DATASET_jpg/results" --dataset ucf101 --model resnet --model_depth 50 --n_classes 5 --batch_size 32 --n_threads 4 --checkpoint 5

for fine tunning
python main.py --root_path "G:/My Drive/Segmented_FYP_DATA_jpg" --video_path . --annotation_path "G:/My Drive/Segmented_FYP_DATA_jpg/dataset.json" --result_path "G:/My Drive/Segmented_FYP_DATA_jpg/results" --dataset ucf101 --model resnet --model_depth 50 --n_classes 5 --n_pretrain_classes 700 --pretrain_path "G:/My Drive/Segmented_FYP_DATA_jpg/models/r3d50_K_200ep.pth" --ft_begin_module fc --batch_size 32 --n_threads 4 --checkpoint 5 --n_epochs 30