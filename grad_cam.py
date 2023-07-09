import cv2

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch.nn as nn

import cv2
import numpy as np
import torch
import iresnet
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from scrfd import SCRFD
from utils import norm_crop

if __name__ == '__main__':
    # init face detection
    detector = SCRFD(model_file='assets/det_10g.onnx')
    detector.prepare(0, det_thresh=0.5, input_size=(160, 160))

    # model-1
    net1 = iresnet.iresnet50()
    weight = 'assets/w600k_r50.pth'
    net1.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
    net1.eval().cuda()

    # model-2
    net2 = iresnet.iresnet100()
    weight = 'assets/glint360k_r100.pth'
    net2.load_state_dict(torch.load(weight, map_location=torch.device('cuda')))
    net2.eval().cuda()

    im_a = cv2.imread('output/001_2.png')
    im_v = cv2.imread('images/002/1.png')

    bboxes, kpss = detector.detect(im_a, max_num=1)
    att_img, M = norm_crop(im_a, kpss[0], image_size=112)

    bboxes, kpss = detector.detect(im_v, max_num=1)
    vic_img, _ = norm_crop(im_v, kpss[0], image_size=112)

    att_img = att_img[:, :, ::-1]  # BGR ==> RGB
    vic_img = vic_img[:, :, ::-1]

    bboxes, kpss = detector.detect(att_img, max_num=1)

    # get victim feature
    vic_img = torch.Tensor(vic_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    vic_img.div_(255)
    vic_feats = net1.forward(vic_img)  # Get feature vector

    # process input
    att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    att_img.div_(255)
    att_img_ = att_img.clone()
    att_img.requires_grad = True


    net1.zero_grad()
    adv_images = att_img.clone()

    # get adv feature
    adv_feats = net1.forward(adv_images)

    # caculate loss and backward
    loss = torch.mean(torch.square(adv_feats - vic_feats)) # MSELoss
    adv_feats = adv_feats.squeeze()
    vic_feats = vic_feats.squeeze()
    target = torch.tensor(1).cuda()
    loss.backward(retain_graph=True)

    grad = att_img.grad.data.clone()
    grad = grad.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    g_max, g_min = grad.max(), grad.min()
    eps = 1e-7
    grad = (grad - g_min) / (g_max - g_min + eps)
    cam = np.uint8(grad * 255)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    att_img = att_img.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    att_img = np.uint8(att_img * 255)
    result = 0.3 * heatmap + 0.7 * att_img
    result = np.uint8(result)

    mask = []
    avg_grad = []
    gray = cv2.cvtColor(cam, cv2.COLOR_RGB2GRAY)
    for i in range(5):
        sub_mask = np.zeros_like(gray)
        x, y = int(kpss[0][i][0]), int(kpss[0][i][1])
        x1 = max(0, x - 16)
        x2 = min(x + 16, 112)
        y1 = max(0, y - 16)
        y2 = min(y + 16, 112)
        sub_mask[x1:x2, y1:y2] = 1
        mask.append(sub_mask)
        area = (gray * mask[i]).sum()
        sub_avg_grad = area / mask[i].sum()
        avg_grad.append(sub_avg_grad)

    # 标出特征点
    for i in range(5):
        x, y = int(kpss[0][i][0]), int(kpss[0][i][1])
        cv2.circle(result, (x, y), 1, (255, 0, 0), 1)

    # 标出Top-5点的位置
    # gray = cv2.cvtColor(grad, cv2.COLOR_RGB2GRAY)
    # x, y, max_grad = [], [], []
    # for _ in range(5):
    #     max = -1
    #     max_i, max_j = 0, 0
    #     for i in range(112):
    #         for j in range(112):
    #             if gray[i][j] in max_grad:
    #                 continue
    #             if gray[i][j] > max:
    #                 max = gray[i][j]
    #                 max_i = i
    #                 max_j = j
    #     x.append(max_i)
    #     y.append(max_j)
    #     max_grad.append(max)
    #
    # for i in range(5):
    #     xx, yy = x[i], y[i]
    #     cv2.circle(result, (xx, yy), 1, (255, 0, 0), 1)

    def show_pic(x):
        plt.imshow(x)
        pylab.show()
    show_pic(heatmap)
    show_pic(result)
