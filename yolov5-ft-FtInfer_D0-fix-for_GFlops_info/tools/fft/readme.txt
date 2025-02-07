1.进入coco-ft.py，配置好coco路径
    coco_root = '/home/liu/datas/coco2017'
    coco_path = coco_root + '/train/images'#图像（仅用于可视化调试）
    labels_path = coco_root + '/train/labels' #输出
    coco_json = coco_root + '/annotations' #输入
2.运行coco-ft.py,结果会生成到labels_path下面的['train2017', 'val2017']
3.将生成的coco_path/train2017 拷贝到训练数据集和images平行的文件夹  注意原来的labels文件换成labels-0
  将生成的coco_path/val2017 拷贝到验证数据集和images平行的文件夹  注意原来的labels文件换成labels-0
  ....
4.运行yolov5-ft工程的train.py