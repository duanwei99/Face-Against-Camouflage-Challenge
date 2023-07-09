import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import glob
import cv2
import numpy as np
import torch
import iresnet
import os
from tqdm import tqdm

from scrfd import SCRFD
from utils import norm_crop

class Tool:
    def __init__(self):
        # SCRFD
        detector = SCRFD(model_file='assets/det_10g.onnx')
        detector.prepare(0, det_thresh=0.5, input_size=(160, 160))
        self.detector = detector

        # model-1
        net1 = iresnet.iresnet50()
        weight = 'assets/w600k_r50.pth'
        net1.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
        net1.eval().cuda()
        self.net1 = net1

        # model-2
        net2 = iresnet.iresnet100()
        weight = 'assets/glint360k_r100.pth'
        net2.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
        net2.eval().cuda()
        self.net2 = net2

        self.score_list = []

tool = Tool()
paths = glob.glob('/data1/duanwei/some-resources/2022/0425/images/001/0.png')
for path in paths:
    img_test = cv2.imread(path)
    bboxes, kpss = tool.detector.detect(img_test, max_num=1)
    for i in range(5):
            cv2.circle(img_test, (int(kpss[0][i][0]), int(kpss[0][i][1])), 2, (0, 0, 255), -1)
    cv2.circle(img_test, (int(bboxes[0][2]), int(bboxes[0][3])), 4, (255, 0, 0), -1)
    cv2.circle(img_test, (int(bboxes[0][0]), int(bboxes[0][1])), 4, (255, 0, 0), -1)
    img_test[img_test < 0] = 0
    img_test = img_test.astype(np.uint8)
    cv2.imshow('img',img_test)
    cv2.imwrite('./test.png',img_test)
    cv2.waitKey(0)
    cv2.destroyWindow('img')



















# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
#
# def show(img, title='img'):
#     plt.imshow(img)
#     plt.title(title)
#     pylab.show()
#
# im1 = cv2.imread('output/001_2.png')
# im2 = cv2.imread('images/001/0.png')
#
# mask = im1 - im2
# mask[mask != 0] = 255
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# show(mask, 'Attack area')
#
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
# # https://www.freesion.com/article/1420638551/
#
# x, y, w, h = stats[1][0], stats[1][1], stats[1][2], stats[1][3]
# mask = cv2.rectangle(mask, (x, y), (x+w, y+h), color=127, thickness=1)
# show(mask, 'Connected domain with bounding rectangle')
# print(w*h)