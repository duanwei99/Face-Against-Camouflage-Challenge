import numpy as np
import cv2
from skimage import transform as trans
import os,sys
from pathlib import Path

arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041] ], dtype=np.float32 )

def estimate_norm(lmk, image_size):
    assert lmk.shape==(5,2)
    tform = trans.SimilarityTransform()
    _src = float(image_size)/112 * arcface_src
    tform.estimate(lmk, _src)
    M = tform.params[0:2,:]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
    return warped, M

log_dir = Path('./')
class logSaver():
    def __init__(self, logname):
        self.logname = logname
        sys.stdout.flush()
        sys.stderr.flush()
        if self.logname == None:
            self.logpath_out = os.devnull
            self.logpath_err = os.devnull
        else:
            self.logpath_out = logname + "_out.log"
            self.logpath_err = logname + "_err.log"
        self.logfile_out = os.open(self.logpath_out, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
        self.logfile_err = os.open(self.logpath_err, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)

    def __enter__(self):
        self.orig_stdout = os.dup(1)
        self.orig_stderr = os.dup(2)
        self.new_stdout = os.dup(1)
        self.new_stderr = os.dup(2)
        os.dup2(self.logfile_out, 1)
        os.dup2(self.logfile_err, 2)
        sys.stdout = os.fdopen(self.new_stdout, 'w')
        sys.stderr = os.fdopen(self.new_stderr, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(self.orig_stdout, 1)
        os.dup2(self.orig_stderr, 2)
        os.close(self.orig_stdout)
        os.close(self.orig_stderr)

        os.close(self.logfile_out)
        os.close(self.logfile_err)