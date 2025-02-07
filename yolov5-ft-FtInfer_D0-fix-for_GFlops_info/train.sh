#!/bin/bash

source /home/LIESMARS/2019286190105/anaconda3/bin/activate
conda activate torch18

# python -m torch.distributed.launch --nproc_per_node 2 train.py --weights weights/yolov5x.pt --data data/hrsc2016.yaml \
#    --cfg models/yolov5x-hrsc2016.yaml --batch 32 --epochs 100 --imgsz 768 --device 0,1 --single --noautoanchor --workers 8 --noval


# python -m torch.distributed.launch --nproc_per_node 2 train.py --weights weights/hrsc2016-88-l-768-fold2.pt --data data/hrsc2016.yaml \
#    --cfg models/yolov5l-hrsc2016.yaml --batch 64 --epochs 100 --imgsz 768 --device 0,1 --single --noautoanchor --workers 8

# dota
# python -m torch.distributed.launch --nproc_per_node 2 train.py --weights weights/yolov5m.pt --data data/dota.yaml \
#    --cfg models/yolov5m-dota.yaml --batch 128 --epochs 300 --imgsz 640 --device 0,1 --noautoanchor --workers 8 --noval
python train.py --weights paperdatas/train/4.5/hrsc-768-se-aug-ms-fold2/weights/best.pt --data data/hrsc2016.yaml --cfg models/yolov5m-se-hrsc2016.yaml --batch 8 --epochs 200 --imgsz 768  --noautoanchor --workers 8 --hyp data/hyps/hyp.hrsc2016.yaml --fold 1

