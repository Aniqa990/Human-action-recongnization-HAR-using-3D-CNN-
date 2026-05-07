python util_scripts/generate_video_jpgs.py

python prepare_annotation.py

mkdir "G:\My Drive\SEG_DATASET_jpg\results"

python main.py --root_path "G:/My Drive/DATASET_jpg" --video_path . --annotation_path "G:/My Drive/DATASET_jpg/dataset.json" --result_path "G:/My Drive/DATASET_jpg/results" --dataset ucf101 --model resnet --model_depth 50 --n_classes 5 --batch_size 32 --n_threads 4 --checkpoint 5

for fine tunning
python main.py --root_path "G:/My Drive/Segmented_FYP_DATA_jpg" --video_path . --annotation_path "G:/My Drive/Segmented_FYP_DATA_jpg/dataset.json" --result_path "G:/My Drive/Segmented_FYP_DATA_jpg/results" --dataset ucf101 --model resnet --model_depth 50 --n_classes 5 --n_pretrain_classes 700 --pretrain_path "G:/My Drive/Segmented_FYP_DATA_jpg/models/r3d50_K_200ep.pth" --ft_begin_module fc --batch_size 32 --n_threads 4 --checkpoint 5 --n_epochs 30

---

step 01:

python step1_generate_frames.py --video_root "G:/My Drive/Augmented_FYP_Data" --jpg_root "G:/My Drive/FYP_DATA_jpg_raw" --n_workers 8

Step 02:

python step2_prepare_annotation.py --jpg_root "G:/My Drive/FYP_DATA_jpg_raw" --output "G:/My Drive/FYP_DATA_jpg_raw/dataset.json" --val_split 0.2

for mobilenet:
python step3_train_mobilenet.py --jpg_root "H:/My Drive/FYP_DATA_jpg_raw" --annotation "H:/My Drive/FYP_DATA_jpg_raw/dataset.json" --result_path "H:/My Drive/FYP_DATA_jpg_raw/results_mobilenet" --n_epochs 30 --batch_size 16 --n_workers 8 --resume_path "H:/My Drive/FYP_DATA_jpg_raw/results_mobilenet/save_7.pth"

for X3D:
python step3_train_x3d.py --jpg_root "G:/My Drive/FYP_DATA_jpg_raw" --annotation "G:/My Drive/FYP_DATA_jpg_raw/dataset.json" --result_path "G:/My Drive/FYP_DATA_jpg_raw/results_x3d" --n_epochs 30 --batch_size 16 --n_workers 8 

python step3_train_x3d.py --jpg_root "H:/My Drive/FYP_DATA_jpg_raw" --annotation "H:/My Drive/FYP_DATA_jpg_raw/dataset.json" --result_path "H:/My Drive/FYP_DATA_jpg_raw/results_x3d" --n_epochs 30 --batch_size 16 --n_workers 8 --resume_path "H:/My Drive/FYP_DATA_jpg_raw/results_x3d/save_5.pth"


for testing:

step 1:
python step1_generate_frames.py --video_root "H:\My Drive\FYP_TEST_VIDEOS" --jpg_root "H:\My Drive\FYP_TEST_JPG" --n_workers 8

step 03:
python step4_test_x3d.py --model_path "H:/My Drive/FYP_DATA_jpg_raw/results_x3d/best_model.pth" --jpg_root "H:/My Drive/FYP_TEST_JPG" --annotation "H:/My Drive/FYP_TEST_JPG/dataset.json" --batch_size 16 --n_frames 16 --img_size 160

# Test X3D
python step4_test_x3d.py --model_path "H:/My Drive/FYP_DATA_jpg_raw/results_x3d/best_model.pth" --jpg_root "H:/My Drive/FYP_TEST_JPG"

# Test MobileNet (just change the path!)
python step4_test_x3d.py --model_path "H:/My Drive/FYP_DATA_jpg_raw/results_mobilenet/best_model.pth" --jpg_root "H:/My Drive/FYP_TEST_JPG" --model_type mobilenet