import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


OUTPUT_PATH = '/data/vision/phillipi/gen-models/svg/dataset/omnipush_black_future'
LENGTH = 12
DSIZE = (64, 64)
N_TEST_PER_SHAPE = 20
SPLIT = ['train', 'test']


train_dirs = glob.glob(os.path.join(OUTPUT_PATH, 'train/**/*'))
test_dirs = glob.glob(os.path.join(OUTPUT_PATH, 'test/**/*'))


def generate_black_img(dir):
    # construct black image
    img_black = np.zeros((64, 64, 3))
    for i in range(LENGTH, 60):
        cv2.imwrite(os.path.join(dir, '{}.png').format(i), img_black)


Parallel(n_jobs=10)(delayed(generate_black_img)(fp) for fp in tqdm(train_dirs))
Parallel(n_jobs=10)(delayed(generate_black_img)(fp) for fp in tqdm(test_dirs))
