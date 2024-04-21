CUDA_VISIBLE_DEVICES=3 \
python pretrain.py \
    --name VOC_PRETRAIN \
    --data_folds_path data_preparation/split-10-30-30-30 \
    --batch_size_per_gpu 24 \
    --voc_path /home/wonjun/data/voc2012/VOCdevkit/VOC2012 \
    --pretrained_dir pretrained_weights/pretrain.ckpt \
    --output_dir outputs/test/pretrain \