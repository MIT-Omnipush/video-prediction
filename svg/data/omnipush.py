import os
import io
import glob
from scipy.misc import imresize
import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread


class OmniPush(object):

    """Data Handler that loads omni-push data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64,
                 use_action=False):
        self.root_dir = data_root
        self.train = train
        self.use_action = use_action
        self.seq_len = seq_len
        self.image_size = image_size
        self.seed_is_set = False  # multi threaded loading

        if self.train:
            self.data_dir = '%s/train' % self.root_dir
            self.ordered = False
        else:
            self.data_dir = '%s/test' % self.root_dir
            self.ordered = True
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                d = '%s/%s/%s' % (self.data_dir, d1, d2)
                length = len(glob.glob('%s/*.png' % d))
                if length >= seq_len:
                    self.dirs.append(d)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return len(self.dirs)

    def get_seq(self, index):
        if self.ordered:
            d = self.dirs[index]
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]

        # Random sample slices of length 12 in sequence
        t_start = 0
        if self.train:
            length = len(glob.glob('%s/*.png' % d)) - self.seq_len
            t_start = np.random.randint(length+1)

        # Construct image sequence
        image_seq = []
        for i in range(t_start, t_start+self.seq_len):
            fname = '%s/%d.png' % (d, i)
            im = imread(fname).reshape(1, 64, 64, 3)
            image_seq.append(im/255.)
        image_seq = np.concatenate(image_seq, axis=0)

        if not self.use_action:
            return image_seq
        else:
            action_all = np.load('%s/actions.npy' % (d))
            action_seq = action_all[t_start:t_start+self.seq_len-1]
            assert image_seq.shape[0] == action_seq.shape[0] + 1
            # Note! Just output pushing direction!
            return (image_seq, action_seq[:, -1:])

    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq(index)
