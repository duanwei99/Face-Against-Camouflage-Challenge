import glob

import cv2
import numpy as np
import torch
import iresnet
from tqdm import tqdm

from scrfd import SCRFD
from utils import norm_crop, logSaver

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
        self.sim1 = []
        self.sim2 = []

    @torch.no_grad()
    def inference(self, net, img):
        bboxes, kpss = self.detector.detect(img, max_num=1)
        if bboxes.shape[0] == 0:
            return None
        bbox = bboxes[0]
        kp = kpss[0]
        aimg = norm_crop(img, kp, image_size=112)

        aimg = aimg[0]
        aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
        aimg = np.transpose(aimg, (2, 0, 1))
        aimg = torch.from_numpy(aimg).unsqueeze(0).float()
        aimg.div_(255).sub_(0.5).div_(0.5)
        # aimg.div_(255)
        aimg = aimg.cuda()
        feat = net(aimg).cpu().numpy().flatten()
        feat /= np.sqrt(np.sum(np.square(feat)))
        return feat, bbox

    def get_connected_domain(self, img1, img2):
        mask = img1 - img2
        mask[mask != 0] = 255
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        return nums, labels, stats, centroids

    def cos_sim(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        cos = np.dot(a, b) / (a_norm * b_norm)
        return cos

    def get_score(self, ori_path, att_path, vic_path, sim_thres, metric='mean'):
        ori_img = cv2.imread(ori_path)
        # net-1
        att_img = cv2.imread(att_path)
        att_feat, att_bbox = self.inference(self.net1, att_img)

        sim1 = []
        for vp in vic_path:
            vic_img = cv2.imread(vp)
            vic_feat, vic_bbox = self.inference(self.net1, vic_img)
            sim1.append(self.cos_sim(att_feat, vic_feat))
        print(sim1)
        if metric == 'mean':
            sim1 = np.mean(sim1)
        elif metric == 'max':
            sim1 = np.max(sim1)
        elif metric == 'min':
            sim1 = np.min(sim1)
        elif metric == 'median':
            sim1 = np.median(sim1)
        if sim1 >= sim_thres:
            nums, labels, stats, _ = self.get_connected_domain(att_img, ori_img)
            att_area = 0
            for i in range(1, nums):
                x, y, w, h = stats[i][0], stats[i][1], stats[i][2], stats[i][3]
                att_area = att_area + w * h
            face_area = (att_bbox[2] - att_bbox[0]) * (att_bbox[3] - att_bbox[1])
            score = (1 - att_area / face_area) * 100
            self.score_list.append(score)
        else:
            self.score_list.append(0)

        # net-2
        att_img = cv2.imread(att_path)
        att_feat, att_bbox = self.inference(self.net2, att_img)


        sim2 = []
        for vp in vic_path:
            vic_img = cv2.imread(vp)
            vic_feat, vic_bbox = self.inference(self.net2, vic_img)
            sim2.append(self.cos_sim(att_feat, vic_feat))
        print(sim2)
        if metric == 'mean':
            sim2 = np.mean(sim2)
        elif metric == 'max':
            sim2 = np.max(sim2)
        elif metric == 'min':
            sim2 = np.min(sim2)
        elif metric == 'median':
            sim2 = np.median(sim2)

        if sim2 >= sim_thres:
            nums, labels, stats, _ = self.get_connected_domain(att_img, ori_img)
            att_area = 0
            for i in range(1, nums):
                x, y, w, h = stats[i][0], stats[i][1], stats[i][2], stats[i][3]
                att_area = att_area + w * h
            face_area = (att_bbox[2] - att_bbox[0]) * (att_bbox[3] - att_bbox[1])
            score = (1 - att_area / face_area) * 100
            self.score_list.append(score)
        else:
            self.score_list.append(0)

        self.sim1.append(sim1)
        print(sim1)
        self.sim2.append(sim2)
        print(sim2)

if __name__ == '__main__':
    with logSaver('e.txt'):
        tool = Tool()
        threshold = 0.3
        for i in tqdm(range(100)):
            if i + 1 < 10:
                x = '00' + str(i + 1)
            elif i + 1 < 100:
                x = '0' + str(i + 1)
            else:
                x = '100'
            ori_img = 'images/' + x + '/0.png'
            att_img = 'output/' + x + '_2.png'
            vic_path = 'same_pic/' + str(i) + '/*.*'
            vic_path = glob.glob(vic_path)
            tool.get_score(ori_img, att_img, vic_path, threshold, metric='mean')
        tool.score_list = np.array(tool.score_list)
        # print(np.median(tool.score_list))
        # print((tool.score_list != 0).sum())
    