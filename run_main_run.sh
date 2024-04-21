CUDA_VISIBLE_DEVICES=3 \
python main_run.py \
    --name VOC_FOLD1 \
    --data_folds_path data_preparation/split-10-30-30-30 \
    --batch_size_per_gpu 8 \
    --voc_path /home/wonjun/data/voc2012/VOCdevkit/VOC2012 \
    --pretrained_dir outputs/split-10-30-30-30/fold1/checkpoint.pth \
    --output_dir outputs/test_split-10-30-30-30/fold2 \
    --num_folds 3 \