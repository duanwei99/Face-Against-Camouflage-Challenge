from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2, json
import numpy as np
import torch
import iresnet
from tqdm import tqdm

from scrfd import SCRFD
from utils import norm_crop

class Recognizer:
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

        # KNN Classifier
        lfw_feats = np.load('lfw_feats.npy')
        face_id = np.load('face_id.npy')
        clf = KNeighborsClassifier(n_neighbors=50, metric='cosine')
        clf.fit(lfw_feats, face_id)
        self.clf = clf
        self.face_id = face_id

        id_map = json.load(open('id_map.json'))
        self.map = id_map

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
        aimg = aimg.cuda()
        feat = net(aimg).cpu().numpy().flatten()
        feat /= np.sqrt(np.sum(np.square(feat)))
        return feat, bbox

    def recognize(self, img, net='iresnet50'):
        img = cv2.imread(img)
        if net == 'iresnet50':
            feat, _ = self.inference(self.net1, img)
        else:
            feat, _ = self.inference(self.net2, img)
        feat = feat.reshape(1, -1)
        distance, idxs = self.clf.kneighbors(feat, return_distance=True)
        conf = 1 - distance
        return conf[0, 0], self.map[self.face_id[idxs[0, 0]]]

if __name__ == '__main__':
    recog = Recognizer()
    im = 'output/001_2.png'
    conf, name = recog.recognize(im)
    print('Result: ' + name)
    print('Conf: {}'.format(conf))