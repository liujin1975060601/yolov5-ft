import cv2
import os
import tqdm

jpg_path = r'E:\PyCharmProject\yolov5_rot_imsize_0627\runs\detect\exp110'

output = r'output_cv_shanxi.mp4'

jpg_list = [os.path.join(jpg_path, i) for i in os.listdir(jpg_path) if os.path.splitext(i)[-1] in ['.jpg', '.png']]
w_h = (1662, 1247)
videoW = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), 15, w_h)
count = 0
for i in tqdm.tqdm(jpg_list):
    frame = cv2.imread(i)
    size = frame.shape # h,w,c
    frame = cv2.resize(frame, w_h)
    # if size[0] != 480 and size[1] != 640:
    #     if size[0]/size[1] != 3/4:
    #         continue
    #     else:
    #         mat = cv2.getRotationMatrix2D((0, 0), 0, 1080/size[1])
    #         frame = cv2.warpAffine(frame, mat, (1080, 1920))
    #
    # frame = frame[90:, 160:, :]
    # cv2.imshow('test', frame)
    # cv2.waitKey(10)
    count += 1
    for j in range(20):
        videoW.write(frame)
    # if count > 5:
    #     break
# os.system(f'explorer {output}')

