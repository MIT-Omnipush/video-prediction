from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import argparse
import errno
import json
import os
import random
import time

import cv2
import numpy as np
import tensorflow as tf

from video_prediction import datasets, models
from video_prediction.utils.ffmpeg_gif import save_gif


parser = argparse.ArgumentParser(description="Run commands")

# CEM
parser.add_argument('--a_dim', type=int, default=2,
                    help="Dimension of action")
parser.add_argument('--n_steps', type=int, default=3,
                    help="Number of steps into future")
parser.add_argument('--n_repeat', type=int, default=10,
                    help="Number of repeated actions")
parser.add_argument('--n_samples', type=int, default=100,
                    help="Number of samples at each step")
parser.add_argument('--n_best', type=int, default=20,
                    help="Number of best actions used to refit")
parser.add_argument('--n_rounds', type=int, default=6,
                    help="Number of iterations for sampling and refitting")
parser.add_argument('--w_smooth_cov', type=float, default=0.5,
                    help="weighted average of previous cov")

# Video Prediction
parser.add_argument("--input_dir", type=str, default='data/dummy_omnipush/val/', help="dataset path")
parser.add_argument("--results_dir", type=str, default='results', help="ignored if output_gif_dir is specified")
parser.add_argument("--results_gif_dir", type=str, default='results/tmp', help="default is results_dir. ignored if output_gif_dir is specified")
parser.add_argument("--results_png_dir", type=str, help="default is results_dir. ignored if output_png_dir is specified")
parser.add_argument("--output_gif_dir", help="output directory where samples are saved as gifs. default is "
                                                "results_gif_dir/model_fname")
parser.add_argument("--output_png_dir", help="output directory where samples are saved as pngs. default is "
                                                "results_png_dir/model_fname")
parser.add_argument("--checkpoint", default='logs/omnipush_no_weight/ours_vae_l1', help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")

parser.add_argument("--mode", type=str, choices=['test'], default='test', help='mode for dataset, val or test.')

parser.add_argument("--dataset", type=str, default='omnipush', help="dataset class name")
parser.add_argument("--dataset_hparams", type=str, default='sequence_length=-1', help="a string of comma separated list of dataset hyperparameters")
parser.add_argument("--model", type=str, default='savp', help="model class name")
parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")

parser.add_argument("--batch_size", type=int, default=20, help="number of samples in batch")
parser.add_argument("--num_samples", type=int, help="number of samples in total (all of them by default)")
parser.add_argument("--num_epochs", type=int, default=1)

parser.add_argument("--num_stochastic_samples", type=int, default=1)
parser.add_argument("--gif_length", type=int, help="default is sequence_length")
parser.add_argument("--fps", type=int, default=12)

parser.add_argument("--gpu_mem_frac", type=float, default=0.9, help="fraction of gpu memory to use")
parser.add_argument("--seed", type=int, default=7)

class CrossEntropyMethod(object):
    """ Optimizes a set of push candidates using the
    cross entropy method.
    Cross entropy method (CEM):
    (1) sample an initial set of candidates
    (2) sort the candidates
    (3) fit a Gaussian to the top P%
    (4) re-sample pushes from the distribution
    (5) repeat steps 2-4 for K iters
    (6) return the best candidate from the final sample set
    """

    def __init__(self, args):
        """
        Parameters
        ----------
        a_dim : int
            Size of action space
        """
        super(CrossEntropyMethod, self).__init__()
        self.a_dim = args.a_dim
        self.n_steps = args.n_steps
        self.n_samples = args.n_samples  # sample M different actions
        self.n_best = args.n_best  # only take the K best actions

        self.w_smooth_cov = args.w_smooth_cov

        self.mean = np.zeros(self.a_dim * self.n_steps)
        self.sigma = np.diag(np.ones(self.a_dim * self.n_steps))

    def fit(self, actions, costs):
        best_actions_ids = sorted(range(len(costs)), key=lambda k: costs[k])
        best_actions = actions[best_actions_ids[:self.n_best]]  # only take the K best actions
        best_actions = best_actions.reshape(self.n_best, self.n_steps * self.a_dim)
        self.sigma = np.cov(best_actions, rowvar=False) + self.w_smooth_cov * self.sigma
        self.mean = np.mean(best_actions, axis=0)

    def sample(self, mean, sigma, size):
        actions = np.random.multivariate_normal(mean, sigma, size=size)
        actions = actions.reshape(size, self.n_steps, self.a_dim)
        return actions

    def act(self):
        actions = self.sample(self.mean, self.sigma, self.n_samples)
        return actions


class Env(object):
    def __init__(self):
        """
        Parameters
        ----------
        init_imgs : python list of images
            Images used to warm up video prediction model.
        """
        super(Env, self).__init__()
        self.reset()

    def reset(self):
        self.state = (0, 0)

    def step(self, action):
        self.state += action
    
    def eval_cost(self, goal):
        return np.linalg.norm(self.state - goal, ord=2) 


if __name__ == '__main__':
    args = parser.parse_args()
    args.dataset_hparams = args.dataset_hparams.replace('-1', '{}'.format(args.n_steps * args.n_repeat + 1))
    # env = Env()
    cem = CrossEntropyMethod(args)

    # goal = (10, 10)

    # for r in range(args.n_rounds):
    #     costs = []
    #     actions = cem.act()  # actions: (n_samples, n_steps, 2)
    #     actions = np.clip(actions, -1, 1)  # truncate actions to (-1, 1)
    #     for action in actions:  # action: (n_steps, 2)
    #         env.reset()
    #         for i in range(args.n_steps):
    #             env.step(action[i])
    #         costs.append(env.eval_cost(goal))
    #     cem.fit(actions, costs)
    #     print(r, '%.3f' % min(costs), np.sum(cem.mean.reshape((cem.n_steps, cem.a_dim)), axis=0))
    #     print(r, '%.3f' % min(costs), cem.mean.reshape((cem.n_steps, cem.a_dim)))
    
    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    args.results_gif_dir = args.results_gif_dir or args.results_dir
    args.results_png_dir = args.results_png_dir or args.results_dir
    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.checkpoint:
        checkpoint_dir = os.path.normpath(args.checkpoint)
        if not os.path.isdir(args.checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            args.dataset = args.dataset or options['dataset']
            args.model = args.model or options['model']
        try:
            with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                dataset_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")
        args.output_gif_dir = args.output_gif_dir or os.path.join(args.results_gif_dir, os.path.split(checkpoint_dir)[1])
        args.output_png_dir = args.output_png_dir or os.path.join(args.results_png_dir, os.path.split(checkpoint_dir)[1])
    else:
        if not args.dataset:
            raise ValueError('dataset is required when checkpoint is not specified')
        if not args.model:
            raise ValueError('model is required when checkpoint is not specified')
        args.output_gif_dir = args.output_gif_dir or os.path.join(args.results_gif_dir, 'model.%s' % args.model)
        args.output_png_dir = args.output_png_dir or os.path.join(args.results_png_dir, 'model.%s' % args.model)

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    VideoDataset = datasets.get_dataset_class(args.dataset)
    dataset = VideoDataset(
        args.input_dir,
        mode=args.mode,
        num_epochs=args.num_epochs,
        seed=args.seed,
        hparams_dict=dataset_hparams_dict,
        hparams=args.dataset_hparams)

    VideoPredictionModel = models.get_model_class(args.model)
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': dataset.hparams.context_frames,
        'sequence_length': dataset.hparams.sequence_length,
        'eval_length': dataset.hparams.eval_length,
        'repeat': dataset.hparams.time_shift,
    })
    model = VideoPredictionModel(
        mode=args.mode,
        hparams_dict=hparams_dict,
        hparams=args.model_hparams)

    sequence_length = model.hparams.sequence_length
    context_frames = model.hparams.context_frames

    future_length = sequence_length - context_frames
    eval_length = model.hparams.eval_length
    if eval_length != -1:
        future_length = eval_length

    if args.num_samples:
        if args.num_samples > dataset.num_examples_per_epoch():
            raise ValueError('num_samples cannot be larger than the dataset')
        num_examples_per_epoch = args.num_samples
    else:
        num_examples_per_epoch = dataset.num_examples_per_epoch()
    if num_examples_per_epoch % args.batch_size != 0:
        raise ValueError('batch_size should evenly divide the dataset size %d' % num_examples_per_epoch)

    inputs = dataset.make_batch(args.batch_size)
    input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}
    with tf.variable_scope(''):
        model.build_graph(input_phs)

    for output_dir in (args.output_gif_dir, args.output_png_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(args), sort_keys=True, indent=4))
        with open(os.path.join(output_dir, "dataset_hparams.json"), "w") as f:
            f.write(json.dumps(dataset.hparams.values(), sort_keys=True, indent=4))
        with open(os.path.join(output_dir, "model_hparams.json"), "w") as f:
            f.write(json.dumps(model.hparams.values(), sort_keys=True, indent=4))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.graph.as_default()

    model.restore(sess, args.checkpoint)
    
    sample_ind = 0
    
    lowest_cost = 1e10
    best_action = None
    best_gen_images = None
    
    it = 0
    while True:
        if args.num_samples and sample_ind >= args.num_samples:
            break
        try:
            input_results = sess.run(inputs)
        except tf.errors.OutOfRangeError:
            print("OutOfRangeError")
            break
        print("planning step %d" % it)

        # Set start and end image
        # Image index start from 1, so it=0 corresponds to start_img_1.png
        if not os.path.exists('/home/mcube/yenchen/savp-omnipush/data/start_img_{}.png'.format(it+1)):
            print('Waiting for new input image ...')
        while not os.path.exists('/home/mcube/yenchen/savp-omnipush/data/start_img_{}.png'.format(it+1)):
            time.sleep(0.5)
            pass
        print('Planning action ...')
        start_image = (cv2.imread('/home/mcube/yenchen/savp-omnipush/data/start_img_{}.png'.format(it+1)) / 255.0)
        goal_image = (cv2.imread('/home/mcube/yenchen/savp-omnipush/data/goal_img.png') / 255.0)
        
        feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
        images_ph_key = list(feed_dict.keys())[0]
        actions_ph_key = list(feed_dict.keys())[4]
        
        # Plan longer only for first iteration
        #if it == 1:
        #    args.n_rounds /= 2
        #    args.n_samples /= 2
        #    args.n_best /= 2
                
        # Reuse previous mean if not first iteration
        # if it > 0:            
        #    # init next step's first mean as current step's second mean
        #    prev_mean = cem.mean.copy()
        #    cem = CrossEntropyMethod(args)
        #    remaining_steps = args.n_steps - it
        #    cem.mean[:remaining_steps * args.a_dim] = prev_mean[it * args.a_dim:]
            
        # Sample actions
        for r in range(args.n_rounds):
            start_time = time.time()
            costs = []
            actions = cem.act()  # actions: (n_samples, n_steps, 2)
            actions = np.clip(actions, -1, 1)  # truncate actions to (-1, 1)
            action_radians = np.arctan2(actions[:, :, 0], actions[:, :, 1])  # # Turn action into radians
            
            n_forward_pass = int(args.n_samples / args.batch_size)
            for f in range(n_forward_pass):                
                batch_action_radians = action_radians[f * args.batch_size : (f+1)*args.batch_size]

                tmp_batch_action_radians = []
                for i in range(args.n_steps):
                    tmp_batch_action_radians.append(np.repeat(batch_action_radians[:, i], args.n_repeat))
                cat_action_radians = np.concatenate(tmp_batch_action_radians)
                # import pdb; pdb.set_trace()
                batch_action_radians = cat_action_radians.reshape((args.batch_size, -1))
                batch_action_radians = np.repeat(batch_action_radians[:, :, np.newaxis], 4, axis=2) # (args.batch_size, args.steps * args.repeat, 4)
                
                # Change first 2 input images to current image
                feed_dict[images_ph_key][:, :2] = start_image
                
                # Change action input
                feed_dict[actions_ph_key] = batch_action_radians
                
                # Foresee the future
                gen_images = sess.run(model.outputs['gen_images'], feed_dict=feed_dict)
                
                # Eval the cost
                # import pdb; pdb.set_trace()
                batch_cost = np.sum(((goal_image[32:, :, :] - gen_images[:, -1, 32:, :, :]) ** 2), axis=(1, 2, 3))
                costs.append(batch_cost)
                
                for idx, cost in enumerate(batch_cost):
                    if cost < lowest_cost:
                        lowest_cost = cost
                        best_action = batch_action_radians[idx]
                        best_gen_images = gen_images[idx]

            # save gif
            gen_images_ = best_gen_images[2:]  # (28, 64, 64, 3)
            gen_images_ = (gen_images_ * 255.0).astype(np.uint8)
            # import pdb; pdb.set_trace()
            context_images_ = (feed_dict[images_ph_key][0] * 255.0).astype(np.uint8)
            
            gen_gif_fname = 'best_gen_images.gif'
            context_and_gen_images = list(context_images_[:context_frames]) + list(gen_images_)
            if args.gif_length:
                context_and_gen_images = context_and_gen_images[:args.gif_length]
            save_gif(os.path.join('/home/mcube/yenchen/savp-omnipush/data/', gen_gif_fname),
                     context_and_gen_images, fps=args.fps)
                         
            # save last image
            #best_gen_last_image = (best_gen_images[0][-args.n_repeat * it -1] * 255.0).astype(np.uint8)
            #best_gen_last_image_name = '/home/mcube/yenchen/savp-omnipush/data/best_gen_last_image.png'
            #best_gen_last_image[:32, :, :] = 0
            #cv2.imwrite(best_gen_last_image_name,cv2.cvtColor(best_gen_last_image, cv2.COLOR_RGB2BGR))
            
            costs = np.concatenate(costs)
            
            # Refit Gaussian
            cem.fit(actions, costs)
            cem_mean_reshaped = cem.mean.reshape((cem.n_steps, cem.a_dim))
            mean_radians = np.arctan2(cem_mean_reshaped[:, 0], cem_mean_reshaped[:, 1])
            
            print('---')
            print("round: %d, min cost: %.3f" % (r, min(costs)))
            print("CEM mean: {}".format(mean_radians))
            print("Planning time: {}".format(time.time()-start_time))

        it += 1
        np.save("/home/mcube/yenchen/savp-omnipush/data/cem_output_action.npy", [mean_radians[0]])  # first action
        if it == 3:
            break
        # # only keep the future frames
        # gen_images = gen_images[:, -future_length:]
        # for i, gen_images_ in enumerate(gen_images):
        #     context_images_ = (input_results['images'][i] * 255.0).astype(np.uint8)
        #     gen_images_ = (gen_images_ * 255.0).astype(np.uint8)

        #     gen_images_fname = 'gen_image_%05d_%02d.gif' % (sample_ind + i, 0)
        #     if args.dataset == 'omnipush':
        #         gen_images_fname = input_results['push_name'][i][0].decode('utf-8') + '_%02d.gif' % (0)
        #     context_and_gen_images = list(context_images_[:context_frames]) + list(gen_images_)
        #     if args.gif_length:
        #         context_and_gen_images = context_and_gen_images[:args.gif_length]
        #     save_gif(os.path.join(args.output_gif_dir, gen_images_fname),
        #                 context_and_gen_images, fps=args.fps)

        #     gen_image_fname_pattern = 'gen_image_%%05d_%%02d_%%0%dd.png' % max(2, len(str(len(gen_images_) - 1)))
        #     for t, gen_image in enumerate(gen_images_):
        #         gen_image_fname = gen_image_fname_pattern % (sample_ind + i, 0, t)
        #         if gen_image.shape[-1] == 1:
        #             gen_image = np.tile(gen_image, (1, 1, 3))
        #         else:
        #             gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
        #         cv2.imwrite(os.path.join(args.output_png_dir, gen_image_fname), gen_image)
        # break

        sample_ind += args.batch_size
    print("You safely arrive here, congrats man! :)")
