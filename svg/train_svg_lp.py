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
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter

from IPython import embed
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--a_dim', type=int, default=8, help='dimensionality of action, or a_t')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')

parser.add_argument('--use_action', type=int, default=0, help='if true, train action-conditional model')
parser.add_argument('--schedule_sampling', type=int, default=0, help='if true, turn on schedule sampling')

parser.add_argument('--plot_freq', type=int, default=30, help='number of epochs to save model')
parser.add_argument('--save_freq', type=int, default=60, help='number of epochs to save model')


opt = parser.parse_args()
if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-prior-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f%s-use_action=%d-schedule_sampling=%d' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.prior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name, opt.use_action, opt.schedule_sampling)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)


import models.lstm as lstm_models
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
else:
    opt.a_dim = 0 if not opt.use_action else opt.a_dim
    frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim+opt.a_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    posterior = lstm_models.gaussian_lstm(opt.g_dim+opt.a_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    prior = lstm_models.gaussian_lstm(opt.g_dim+opt.a_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    frame_predictor.apply(utils.init_forget_bias_to_one)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model
    elif opt.image_width == 128:
        import models.dcgan_128 as model
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)

if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) =
    #   log( sqrt(
    #
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)
print("Train sequence: {}, Test sequence: {}".format(len(train_data), len(test_data)))
train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
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

# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    if opt.use_action:
        actions = x[1]
        x = x[0]
    nsample = 20
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])
                h_target = h_target[0].detach()
                if opt.use_action:
                    h = torch.cat([h, actions[i-1].repeat(1, opt.a_dim)], 1)
                    h_target = torch.cat([h_target, actions[i-1].repeat(1, opt.a_dim)], 1)
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                if opt.use_action:
                    h = torch.cat([h, actions[i-1].repeat(1, opt.a_dim)], 1)
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq[s].append(x_in)

    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        # best sequence
        min_mse = 1e7
        for s in range(nsample):
            mse = 0
            for t in range(opt.n_eval):
                mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
            if mse < min_mse:
                min_mse = mse
                min_idx = s

        s_list = [min_idx,
                  np.random.randint(nsample),
                  np.random.randint(nsample),
                  np.random.randint(nsample),
                  np.random.randint(nsample)]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i])
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)
    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)
    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
    utils.save_gif(fname, gifs)

def plot_rec(x, epoch):
    if opt.use_action:
        actions = x[1]
        x = x[0]
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[i-1])
        h_target = encoder(x[i])
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h, _ = h
        h_target, _ = h_target
        h = h.detach()
        h_target = h_target.detach()
        if opt.use_action:
            h = torch.cat([h, actions[i-1].repeat(1, opt.a_dim)], 1)
            h_target = torch.cat([h_target, actions[i-1].repeat(1, opt.a_dim)], 1)
        z_t, _, _= posterior(h_target)
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1))
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i])
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

# --------- helper funtions ------------------------------------
def should(epoch, epochs_per_action):
    if epoch % epochs_per_action == 0:
        return True
    else:
        return False

# --------- training funtions ------------------------------------
def schedule_sampling(x, x_pred, prob):
    # Make sure we are using prediction, not x_pred declared under mse and kld.
    assert x_pred[0, 0, 0, 0] != -1

    # Ensure that eventually, the model is deterministically
    # autoregressive (as opposed to autoregressive with very high probability).
    if prob > 0.99:
        prob = 1
    if torch.rand(1) < prob:
        return x_pred
    else:
        return x


def train(x, epoch):
    frame_predictor.zero_grad()
    posterior.zero_grad()
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # Unpack tuple is use_action is true, where x = (images, actions)
    if opt.use_action:
        actions = x[1]
        x = x[0]

    # Initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()

    mse = 0
    kld = 0
    x_pred = torch.ones(x[0].shape) * -1  # just for schedule sampling, wrong if network encode it
    for i in range(1, opt.n_past+opt.n_future):
        if opt.schedule_sampling and i > opt.n_past:
            prob = float(epoch) / opt.n_epochs  # linear schedule
            x_in = schedule_sampling(x[i-1], x_pred.detach(), prob)
        else:
            x_in = x[i-1]
        h = encoder(x_in)
        h_target = encoder(x[i])[0]
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h = h[0]
        if opt.use_action:
            h = torch.cat([h, actions[i-1].repeat(1, opt.a_dim)], 1)
            h_target = torch.cat([h_target, actions[i-1].repeat(1, opt.a_dim)], 1)

        z_t, mu, logvar = posterior(h_target)
        _, mu_p, logvar_p = prior(h)
        h_pred = frame_predictor(torch.cat([h, z_t], 1))

        x_pred = decoder([h_pred, skip])
        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, mu_p, logvar_p)

    loss = mse + kld*opt.beta
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past)

# --------- training loop ------------------------------------
x_test = next(testing_batch_generator)
# writer = SummaryWriter()
epoch_size = len(train_data) // opt.batch_size
for epoch in range(opt.n_epochs):
    frame_predictor.train()
    posterior.train()
    prior.train()
    encoder.train()
    decoder.train()
    epoch_mse = 0
    epoch_kld = 0

    for i in tqdm(range(epoch_size)):
        time_start = time.time()
        x = next(training_batch_generator)
        # train frame_predictor
        mse, kld = train(x, epoch)
        epoch_mse += mse
        epoch_kld += kld

    print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch, epoch_mse/epoch_size, epoch_kld/epoch_size, epoch*epoch_size*opt.batch_size))
    # writer.add_scalar('data/mse_loss', epoch_mse/epoch_size, epoch)
    # writer.add_scalar('data/kld_loss', epoch_kld/epoch_size, epoch)

    if should(epoch, opt.plot_freq):
        # plot some stuff
        frame_predictor.eval()
        #encoder.eval()
        #decoder.eval()
        posterior.eval()
        prior.eval()

        # Always test the first batch
        plot(x_test, epoch)
        plot_rec(x_test, epoch)

    # Save the model
    if should(epoch, opt.save_freq):
        torch.save({
            'encoder': encoder,
            'decoder': decoder,
            'frame_predictor': frame_predictor,
            'posterior': posterior,
            'prior': prior,
            'opt': opt},
            '%s/model_%d.pth' % (opt.log_dir, epoch))
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)
