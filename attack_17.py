import sys
import os
import os.path as osp
import numpy as np
import datetime
import random
import torch
import glob
import time
import cv2
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import iresnet
from scrfd import SCRFD
from utils_517 import norm_crop, get_max_grad_mask
from utils import logSaver

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt


# TODO: Random seed

class PyFAT:

    def __init__(self, N=10):
        os.environ['PYTHONHASHSEED'] = str(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self.device = torch.device('cpu')
        self.is_cuda = False
        self.num_iter = 400
        self.alpha = 1.0 / 255

    def set_cuda(self):
        self.is_cuda = True
        self.device = torch.device('cuda')
        torch.cuda.manual_seed_all(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def load(self, assets_path):
        detector = SCRFD(model_file=osp.join(assets_path, 'det_10g.onnx'))
        ctx_id = -1 if not self.is_cuda else 0
        detector.prepare(ctx_id, det_thresh=0.5, input_size=(160, 160))
        img_shape = (112, 112)
        # model = iresnet.iresnet50()
        # weight = osp.join(assets_path, 'w600k_r50.pth')
        model = iresnet.iresnet100()
        weight = osp.join(assets_path, 'glint360k_r100.pth')
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval().to(self.device)

        # load face mask
        mask_np = cv2.resize(cv2.imread(osp.join(assets_path, 'mask_1.png')), img_shape) / 255
        mask = torch.Tensor(mask_np.transpose(2, 0, 1)).unsqueeze(0)
        mask = F.interpolate(mask, img_shape).to(self.device)
        self.detector = detector
        self.model1 = model
        self.mask = mask

        model = iresnet.iresnet50()
        weight = osp.join(assets_path, 'w600k_r50.pth')
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval().to(self.device)
        self.model2 = model

    def size(self):
        return 1

    def generate(self, im_a, vic_path, n):
        h, w, c = im_a.shape
        bboxes, kpss = self.detector.detect(im_a, max_num=1)
        if bboxes.shape[0] == 0:
            return None
        att_img, M = norm_crop(im_a, kpss[0], image_size=112)

        vic_feats = []
        for im_v in vic_path:
            im_v = cv2.imread(im_v)
            bboxes, kpss = self.detector.detect(im_v, max_num=1)
            if bboxes.shape[0] == 0:
                return None
            vic_img, _ = norm_crop(im_v, kpss[0], image_size=112)

            att_img = att_img[:, :, ::-1]  # BGR ==> RGB
            vic_img = vic_img[:, :, ::-1]

            # get victim feature
            vic_img = torch.Tensor(vic_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(
                self.device)  # (H, W, C) ==> (B, C, W, H)
            vic_img.div_(255).sub_(0.5).div_(0.5)  # Normalize
            vic_feat = (self.model1.forward(vic_img) + self.model2.forward(vic_img)) / 2 # Get feature vector
            vic_feats.append(vic_feat.detach().cpu().numpy())
        vic_feats = torch.from_numpy(np.array(vic_feats).reshape(len(vic_path), -1)).to(self.device)

        # get kpss of att_img afer norm_crop
        bboxes, kpss = self.detector.detect(att_img, max_num=1)

        # process input
        att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)  # (H, W, C) ==> (B, C, W, H)
        att_img.div_(255).sub_(0.5).div_(0.5)  # Normalize
        att_img_ = att_img.clone()
        att_img.requires_grad = True
        loss_func = nn.CosineEmbeddingLoss()
        cos_sims = []

        def get_cos_sim(a, b):
            a = a.detach().squeeze().cpu().numpy()
            b = b.detach().squeeze().cpu().numpy()
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            cos = np.dot(a, b) / (a_norm * b_norm)
            return cos

        # g = 0
        # decay_factor = 0.8
        for i in range(self.num_iter):
            self.model1.zero_grad()
            self.model2.zero_grad()
            adv_images = att_img.clone()

            # get adv feature
            adv_feats = (self.model1.forward(adv_images) + self.model2.forward(adv_images)) / 2

            # caculate loss and backward
            # loss = torch.mean(torch.square(adv_feats - vic_feats)) # MSELoss
            adv_feats = adv_feats.squeeze().repeat(len(vic_feats), 1)
            vic_feats = vic_feats.to(self.device)
            target = torch.tensor([1 for _ in range(len(vic_feats))]).to(self.device)
            loss = loss_func(adv_feats, vic_feats, target)
            # loss = 0
            # for j in range(len(vic_feats)):
            #     loss = loss + torch.sum(torch.abs(adv_feats - vic_feats[j]).pow(3))
            loss.backward(retain_graph=True)

            # choose mask
            if i == 0:
                grad = att_img.grad.data.clone()
                mask = get_max_grad_mask(grad, kpss, extend_size=16)
                mask = torch.from_numpy(mask).to(self.device)

            # I-FGSM
            grad = att_img.grad.data.clone()
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            sum_grad = grad
            # att_img.data = att_img.data - torch.sign(sum_grad) * self.alpha * (1 - self.mask)
            att_img.data = att_img.data - torch.sign(sum_grad) * self.alpha * mask
            att_img.data = torch.clamp(att_img.data, -1.0, 1.0)
            att_img = att_img.data.requires_grad_(True)

            # MI-FGSM
            # grad = att_img.grad.data.clone()
            # g = decay_factor * g + grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
            # att_img.data = att_img.data - torch.sign(g) * self.alpha * mask
            # att_img.data = torch.clamp(att_img.data, -1.0, 1.0)
            # att_img = att_img.data.requires_grad_(True)

        adv_images = att_img.clone()
        adv_feats = (self.model1.forward(adv_images) + self.model2.forward(adv_images)) / 2
        sims = []
        for j in range(len(vic_feats)):
            sim = get_cos_sim(adv_feats, vic_feats[j])
            sims.append(sim)
        print(sims)
        # get diff and adv img
        diff = att_img - att_img_
        diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
        diff = cv2.warpAffine(src=diff, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
        diff_bgr = diff[:, :, ::-1]
        adv_img = im_a + diff_bgr
        return adv_img, sims

    def mi_fgsm_attack(self, att_img, v_feat, decay_factor):
        g = 0
        for i in range(self.num_iter):
            self.model1.zero_grad()
            self.model2.zero_grad()
            adv_images = att_img.clone()

            # get adv feature
            adv_feats = (self.model1.forward(adv_images) + self.model2.forward(adv_images)) / 2

            # caculate loss and backward
            loss = torch.mean(torch.square(adv_feats - v_feat))  # MSELoss
            loss.backward(retain_graph=True)

            grad = att_img.grad.data.clone()
            g = decay_factor * g + grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            att_img.data = att_img.data - torch.sign(g) * self.alpha * (1 - self.mask)
            att_img.data = torch.clamp(att_img.data, -1.0, 1.0)
            att_img = att_img.data.requires_grad_(True)
        return att_img

def main(args):
    # make directory
    save_dir = args.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tool = PyFAT()
    if args.device == 'cuda':
        tool.set_cuda()
    tool.load('assets')

    sims = []
    for idname in tqdm(range(1, 101)):
        str_idname = "%03d" % idname
        iddir = osp.join('images', str_idname)
        att = osp.join(iddir, '0.png')
        vic_path = 'same_pic/' + str(idname-1) + '/'
        vic_path = glob.glob(vic_path + '*.*')
        origin_att_img = cv2.imread(att)

        ta = datetime.datetime.now()
        adv_img, sim = tool.generate(origin_att_img, vic_path, 0)
        sims.append(sim)
        if adv_img is None:
            adv_img = origin_att_img
        tb = datetime.datetime.now()
        # print( (tb-ta).total_seconds() )
        save_name = '{}_2.png'.format(str_idname)

        cv2.imwrite(save_dir + '/' + save_name, adv_img)

    with open('attack_similarity.txt', 'w') as f:
        for x in sims:
            f.write(str(x))
            f.write('\n')


if __name__ == '__main__':
    with logSaver('e.txt'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', help='output directory', type=str, default='output/')
        parser.add_argument('--device', help='device to use', type=str, default='cuda')
        args = parser.parse_args()
        main(args)

