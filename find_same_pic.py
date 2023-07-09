import argparse

import cv2
import os, shutil, json
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os.path as osp
import numpy as np
import torch
import iresnet

from skimage import transform as trans
from scrfd import SCRFD
from utils import norm_crop
from tqdm import tqdm

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos

@torch.no_grad()
def inference(detector, net, img):
    bboxes, kpss = detector.detect(img, max_num=1)
    if bboxes.shape[0]==0:
        return None
    bbox = bboxes[0]
    kp = kpss[0]
    aimg = norm_crop(img, kp, image_size=112)
    aimg = aimg[0]
    aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
    aimg = np.transpose(aimg, (2, 0, 1))
    aimg = torch.from_numpy(aimg).unsqueeze(0).float()
    aimg.div_(255).sub_(0.5).div_(0.5)
    aimg = aimg.cuda()
    feat = net(aimg).cpu().numpy().flatten()
    feat /= np.sqrt(np.sum(np.square(feat)))
    return feat, bbox

if __name__ == "__main__":

    # init face detection
    detector = SCRFD(model_file = 'assets/det_10g.onnx')
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

    paths = []
    lfw_map = {}
    label = 0
    face_id = []
    id_map = []
    for f_path, dir_name, f_names in os.walk(r'lfw/'):
        if bool(f_path[4:]):
            paths.extend([f_path + '/' + x for x in f_names])
            tmp = [label for _ in range(len(f_names))]
            label += 1
            face_id.extend(tmp)
            lfw_m = {
                f_path[4:]: {
                    'image_names': f_names,
                    'face_id': tmp
                }
            }
            lfw_map.update(lfw_m)

            id_map.append(f_path[4:])
    face_id = np.array(face_id)
    np.save('face_id.npy', face_id)
    with open('id_map.json', 'w') as obj:
        json.dump(id_map, obj)

    # get feats of lfw
    # lfw_feats = []
    # for p in tqdm(paths):
    #     img = cv2.imread(p)
    #     feat, bbox = inference(detector, net1, img)
    #     lfw_feats.append(feat)
    # np.save('lfw_feats.npy', np.array(lfw_feats))
    lfw_feats = np.load('lfw_feats.npy')

    # get victim feats
    # vic_feats = []
    # for i in tqdm(range(100)):
    #     if i + 1 < 10:
    #         x = '00' + str(i + 1)
    #     elif i + 1 < 100:
    #         x = '0' + str(i + 1)
    #     else:
    #         x = '100'
    #     vic_img = 'images/' + x + '/1.png'
    #     vic_img = cv2.imread(vic_img)
    #     feat, bbox = inference(detector, net1, vic_img)
    #     vic_feats.append(feat)
    # np.save('vic_feats.npy', np.array(vic_feats))
    vic_feats = np.load('vic_feats.npy')

    similar_image = []
    threshold = 0.8
    if os.path.exists('same_pic'):
        shutil.rmtree('same_pic')
    os.mkdir('same_pic')
    vic_names = []
    for index, feat in enumerate(tqdm(vic_feats)):
        os.mkdir('same_pic/' + str(index))
        if index + 1 < 10:
            x = '00' + str(index + 1)
        elif index + 1 < 100:
            x = '0' + str(index + 1)
        else:
            x = '100'
        vic_img = 'images/' + x + '/1.png'
        import glob

        for i in range(len(lfw_feats)):
            if cos_sim(feat, lfw_feats[i, :]) >= threshold and osp.basename(paths[i])[:-9] not in vic_names:
                vic_names.append(osp.basename(paths[i])[:-9])

    for index, vic_name in enumerate(vic_names):
        all_image_paths = glob.glob('lfw/' + vic_name + '/*.*')
        for aip in all_image_paths:
            shutil.copyfile(aip, 'same_pic/' + str(index) + '/' + os.path.basename(aip))
    print()