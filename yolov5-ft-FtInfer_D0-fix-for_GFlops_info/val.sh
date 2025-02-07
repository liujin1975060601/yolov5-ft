#!/bin/bash

source /home/LIESMARS/2019286190105/anaconda3/bin/activate
conda activate torch18
# source /root/anaconda3/bin/activate
# conda activate pytorch

# hrsc
# python val.py --data data/hrsc2016.yaml --weights weights/best.pt  --batch 16 --imgsz 768 --conf 0.001 --iou 0.1 --ab_thres 7.0 --fold 2

# ucas
# python val.py --data data/ucas.yaml --weights weights/ucas-640-dir-88.89/weights/best.pt  --batch 8 --imgsz 640 --conf 0.001 --iou 0.1 --ab_thres 3.0 
# ucas big
# python val_big.py --data data/ucas.yaml --weights weights/ucas512-m-91.7-640-fold2.pt  --batch 8 --imgsz 640 --conf 0.001 --iou 0.1 --ab_thres 3.0 --fold 2 \
#     --subsize 512 --overlap 100


# dota big
python val_big.py --data data/dota.yaml --weights weights/dota1-28/weights/best.pt  --batch 16 --imgsz 800 --conf 0.01 --iou 0.1 --ab_thres 3.0 \
    --subsize 768 --overlap 200

# dota
# python val.py --data data/dota.yaml --weights runs/train/exp5/weights/best.pt  --batch 4 --imgsz 768 --conf 0.001 --iou 0.1 --ab_thres 3.0 --fold 2