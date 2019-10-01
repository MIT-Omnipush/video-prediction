import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np
from scipy.ndimage.filters import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=28, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=5, help='number of samples')
parser.add_argument('--plot_train', type=int, default=0, help='if true, also predict training data')
parser.add_argument('--N', type=int, default=256, help='number of samples')
parser.add_argument('--use_action', type=int, default=0, help='if true, train action-conditional model')
parser.add_argument('--a_dim', type=int, default=8, help='dimensionality of action, or a_t')

opt = parser.parse_args()
os.makedirs('%s' % opt.log_dir, exist_ok=True)


opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# ---------------- load the models  ----------------
tmp = torch.load(opt.model_path)
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
prior = tmp['prior']
frame_predictor.eval()
prior.eval()
posterior.eval()
encoder = tmp['encoder']
decoder = tmp['decoder']
encoder.train()
decoder.train()
frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
prior.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()

# ---------------- set the options ----------------
opt.dataset = tmp['opt'].dataset
opt.last_frame_skip = tmp['opt'].last_frame_skip
opt.channels = tmp['opt'].channels
opt.image_width = tmp['opt'].image_width

print(opt)


# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.num_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.num_threads,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True)


def get_batch_generator(data_loader):
    while True:
        for sequence in data_loader:
            if not opt.use_action:
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch
            else:
                images, actions = sequence
                images = utils.normalize_data(opt, dtype, images)
                actions = utils.sequence_input(actions.transpose_(0, 1), dtype)
                yield images, actions

training_batch_generator = get_batch_generator(train_loader)
testing_batch_generator = get_batch_generator(test_loader)

# --------- eval funtions ------------------------------------
def make_gifs(x, idx, names):
    all_gt = x.copy()
    if opt.use_action:
        actions = x[1]
        x = x[0]
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_eval):
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= posterior(h_target) # take the mean
        if i < opt.n_past:
            if not opt.use_action:
                frame_predictor(torch.cat([h, z_t], 1))
            else:
                frame_predictor(torch.cat(
                    [h, z_t, actions[i-1].repeat(1, opt.a_dim)], 1))
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            if not opt.use_action:
                h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            else:
                h_pred = frame_predictor(torch.cat(
                    [h, z_t, actions[i-1].repeat(1, opt.a_dim)], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)

    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    progress = progressbar.ProgressBar(maxval=nsample).start()
    all_gen = []
    for s in range(nsample):
        progress.update(s+1)
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                if not opt.use_action:
                    frame_predictor(torch.cat([h, z_t], 1))
                else:
                    frame_predictor(torch.cat(
                        [h, z_t, actions[i-1].repeat(1, opt.a_dim)], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                if not opt.use_action:
                    h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                else:
                    h = frame_predictor(torch.cat(
                        [h, z_t, actions[i-1].repeat(1, opt.a_dim)], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)

    progress.finish()
    utils.clear_progressbar()


    ###### ssim ######
    for i in range(opt.batch_size):
        for s in range(nsample):
            imgs = []
            for t in range(opt.n_eval):
                img = all_gen[s][t][i].cpu().transpose(0,1).transpose(1,2).clamp(0,1).numpy()
                img = (img * 255).astype(np.uint8)
                imgs.append(img)

            fname = '%s/%s_%d.gif' % (opt.log_dir, names[i], s)
            utils.save_gif_IROS_2019(fname, imgs)

        # save ground truth
        imgs_gt = []
        for t in range(opt.n_eval):
            img = all_gt[t][i].cpu().transpose(0,1).transpose(1,2).clamp(0,1).numpy()
            img = (img * 255).astype(np.uint8)
            imgs_gt.append(img)
        fname = '%s/%s_gt.gif' % (opt.log_dir, names[i])
        utils.save_gif_IROS_2019(fname, imgs_gt)


for i in range(0, opt.N, opt.batch_size):
    print("Ploting examples %d to %d" % (i, (i+opt.batch_size)))
    # plot train
    if opt.plot_train:
        train_x = next(training_batch_generator)
        make_gifs(train_x, i, 'train')

    # plot test
    test_x = next(testing_batch_generator)
    names = test_data.dirs[i:i+opt.batch_size]
    for j in range(len(names)):
        names[j] = names[j].split('/')[-1]
    assert len(names) == opt.batch_size
    print(names)
    make_gifs(test_x, i, names)
