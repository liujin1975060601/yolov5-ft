#!/bin/bash

source /home/LIESMARS/2019286190105/anaconda3/bin/activate
conda activate torch18
# source /root/anaconda3/bin/activate
# conda activate pytorch



# hrsc
# python detect.py --source /home/LIESMARS/2019286190105/datasets/final-master/HRSC/val/images --weights weights/hrsc2016-m-96.8-ms-fold2.pt  \
#      --imgsz 800 800 --conf 0.4 --iou 0.1 --ab_thres 7.0 --fold 2


# ucas big
# python detect_big.py --source /home/LIESMARS/2019286190105/datasets/final-master/UCASALL/UCAS/val/images \
#      --weights weights/ucas512-89-m-640-fold2.pt --imgsz 640 640 --conf 0.25 --iou 0.1 --ab_thres 3.0 --fold 1

# python detect_big.py --source /home/LIESMARS/2019286190105/datasets/final-master/UCASALL/UCAS/val/images \
#      --weights weights/ucas512-m-91.7-640-fold2.pt  --imgsz 640 640 --conf 0.25 --iou 0.1 --ab_thres 3.0 --fold 2
# python detect.py --source /home/LIESMARS/2019286190105/datasets/final-master/UCASALL/UCAS_split/val/images \
#       --weights weights/ucas512-m-91.7-640-fold2.pt --imgsz 640 640 --conf 0.25 --iou 0.1 --ab_thres 3.0 --fold 2


# dota big
python detect_big.py --source /home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA1.0-1.5/val/images \
     --weights weights/dota1-800-dir-36/weights/best.pt --imgsz 800 800 --conf 0.25 --iou 0.1 --ab_thres 3.0

# # dota 
#python detect.py --source /home/LIESMARS/2019286190105/datasets/final-master/DOTA/DOTA768/train/images \
#      --weights weights/best.pt --imgsz 1024 1024 --conf 0.1 --iou 0.2 --ab_thres 1.0 --fold 2 --plot_label