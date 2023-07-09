import os
import os.path as osp
import numpy as np
import datetime
import random
import torch
import glob
import cv2
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import iresnet
from scrfd import SCRFD
from utils import norm_crop, logSaver
# from utils_517 import
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_cos_sim(a, b):
    a = a.detach().squeeze().cpu().numpy()
    b = b.detach().squeeze().cpu().numpy()
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos
# TODO: Random seed

class PyFAT:

    def __init__(self, N=10):
        os.environ['PYTHONHASHSEED'] = str(1)
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self.device = torch.device('cpu')
        self.is_cuda = False
        self.epoch = 3 # 攻击总轮次
        self.num_iter = 50 # 每一轮的迭代次数
        self.num_images = 4 # 每一次迭代选择的图片数量
        self.alpha = (0.5 / 255) / self.epoch

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
        self.detector = detector

        # load face mask
        mask_np = cv2.resize(cv2.imread(osp.join(assets_path, 'mask_520.png')), img_shape) / 255
        mask = torch.Tensor(mask_np.transpose(2, 0, 1)).unsqueeze(0)
        mask = F.interpolate(mask, img_shape).to(self.device)
        self.mask = mask

        # Load IResNet50
        model = iresnet.iresnet50()
        weight = osp.join(assets_path, 'w600k_r50.pth')
        model.load_state_dict(torch.load(weight, map_location=self.device))
        model.eval().to(self.device)
        self.model1 = model

        # Load IResNet100
        model = iresnet.iresnet100()
        weight = osp.join(assets_path, 'glint360k_r100.pth')
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

        # Get feature vectors of all the victims
        vic_feats_ir50 = []
        vic_feats_ir100 = []
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

            # IResNet50
            vic_feat = self.model1.forward(vic_img)
            vic_feats_ir50.append(vic_feat)

            # IResNet100
            vic_feat = self.model2.forward(vic_img)
            vic_feats_ir100.append(vic_feat)

        # get kpss of att_img after norm_crop
        # bboxes, kpss = self.detector.detect(att_img, max_num=1)

        # process input
        att_img = torch.Tensor(att_img.copy()).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)  # (H, W, C) ==> (B, C, W, H)
        att_img.div_(255).sub_(0.5).div_(0.5)  # Normalize
        att_img_ = att_img.clone()

        # Loss function
        loss_func = nn.CosineEmbeddingLoss()

        # Iteration based on "self.epoch"
        for i in range(self.epoch):
            # Random choose "num_images" images of victims
            img_indices = np.random.randint(0, len(vic_path), (self.num_images,))
            vic_feats_ir50_sub = []
            vic_feats_ir100_sub = []
            for x in img_indices:
                vic_feats_ir50_sub.append(vic_feats_ir50[x])
                vic_feats_ir100_sub.append(vic_feats_ir100[x])
            vic_feats_ir50_sub = torch.stack(vic_feats_ir50_sub)
            vic_feats_ir100_sub = torch.stack(vic_feats_ir100_sub)

            # Iteration based on "self.num_iter"
            adv_image = att_img.clone()
            adv_image.requires_grad = True
            for j in tqdm(range(self.num_images)):
                # Attack IResNet50
                adv_image = self.attack_model(adv_image, vic_feats_ir50_sub[j], 'ir50', loss_func,im_a,M,w,h)

                # Attack IResNet100
                adv_image = self.attack_model(adv_image, vic_feats_ir100_sub[j], 'ir100', loss_func,im_a,M,w,h)

        # Calculate cosine similarity
        adv_feats_ir50 = self.model1.forward(vic_img)
        adv_feats_ir100 = self.model2.forward(vic_img)
        sims_ir50 = []
        for i in range(len(vic_path)):
            sim = get_cos_sim(adv_feats_ir50, vic_feats_ir50[i])
            sims_ir50.append(sim)
        sims_ir100 = []
        for i in range(len(vic_path)):
            sim = get_cos_sim(adv_feats_ir100, vic_feats_ir100[i])
            sims_ir100.append(sim)
        print('\n')
        print(sims_ir50)
        print(sims_ir100)

        # get diff and adv img
        diff = adv_image - att_img_
        diff = diff.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
        diff = cv2.warpAffine(src=diff, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
        diff_bgr = diff[:, :, ::-1]
        adv_img = im_a + diff_bgr
        return adv_img, sims_ir50, sims_ir100

    def attack_model(self, adv_image, vic_feats, model_name, loss_func,im_a,M,w,h):
        for i in range(self.num_iter):
            if model_name == 'ir50':
                model = self.model1
            else:
                model = self.model2
            adv_image_ = adv_image.clone()
            model.zero_grad()
            adv_feats = model(adv_image)

            target = torch.tensor([1]).to(self.device)
            loss = loss_func(adv_feats, vic_feats, target)
            loss.backward(retain_graph=True)

            grad = adv_image.grad.data.clone()
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            sum_grad = grad
            adv_image.data = adv_image.data - torch.sign(sum_grad) * self.alpha * (1-self.mask)
            adv_image.data = torch.clamp(adv_image.data, -1.0, 1.0)
            dif = adv_image - adv_image_
            dif = dif.cpu().detach().squeeze().numpy().transpose(1, 2, 0) * 127.5
            dif = cv2.warpAffine(src=dif, M=M, dsize=(w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=0.0)
            dif_bgr = dif[:, :, ::-1]
            adv_img = im_a + dif_bgr



            # adv_image = adv_image.data.requires_grad_(True)
        return adv_image

def main(args):
    # make directory
    save_dir = args.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tool = PyFAT()
    if args.device == 'cuda':
        tool.set_cuda()
    tool.load('assets')

    sims_ir50 = []
    sims_ir100 = []
    for idname in tqdm(range(1, 101)):
        str_idname = "%03d" % idname
        iddir = osp.join('images', str_idname)
        att = osp.join(iddir, '0.png')
        vic_path = 'same_pic/' + str(idname-1) + '/'
        vic_path = glob.glob(vic_path + '*.*')
        origin_att_img = cv2.imread(att)

        ta = datetime.datetime.now()
        adv_img, sim_ir50, sim_ir100 = tool.generate(origin_att_img, vic_path, 0)
        sims_ir50.append(sim_ir50)
        sims_ir100.append(sim_ir50)
        if adv_img is None:
            adv_img = origin_att_img
        tb = datetime.datetime.now()
        # print( (tb-ta).total_seconds() )
        save_name = '{}_2.png'.format(str_idname)

        cv2.imwrite(save_dir + '/' + save_name, adv_img)

    with open('attack_similarity.txt', 'w') as f:
        for i in range(len(sims_ir50)):
            f.write(str(sims_ir50[i]))
            f.write(str(sims_ir100[i]))
            f.write('\n\n')


if __name__ == '__main__':
    with logSaver('e.txt'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', help='output directory', type=str, default='output/')
        parser.add_argument('--device', help='device to use', type=str, default='cuda')
        args = parser.parse_args()
        main(args)