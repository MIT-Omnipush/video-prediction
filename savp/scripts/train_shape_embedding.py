from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import json
import os
import random
import time

import numpy as np
import tensorflow as tf

from video_prediction import datasets
from video_prediction.models.resnet import ShapeEmbeddingModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="either a directory containing subdirectories "
                                                                     "train, val, test, etc, or a directory containing "
                                                                     "the tfrecords")
    parser.add_argument("--val_input_dir", type=str, help="directories containing the tfrecords. default: input_dir")
    parser.add_argument("--logs_dir", default='logs', help="ignored if output_dir is specified")
    parser.add_argument("--output_dir", help="output directory where json files, summary, model, gifs, etc are saved. "
                                             "default is logs_dir/model_fname, where model_fname consists of "
                                             "information from model and model_hparams")
    parser.add_argument("--output_dir_postfix", default="")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--resume", action='store_true', help='resume from lastest checkpoint in output_dir.')

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--dataset_hparams_dict", type=str, help="a json file of dataset hyperparameters")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--model_hparams_dict", type=str, help="a json file of model hyperparameters")

    parser.add_argument("--summary_freq", type=int, default=10000, help="save frequency of summaries (except for image and eval summaries) for train/validation set")
    # parser.add_argument("--image_summary_freq", type=int, default=5000, help="save frequency of image summaries for train/validation set")
    # parser.add_argument("--eval_summary_freq", type=int, default=25000, help="save frequency of eval summaries for train/validation set")
    # parser.add_argument("--accum_eval_summary_freq", type=int, default=100000, help="save frequency of accumulated eval summaries for validation set only")
    parser.add_argument("--progress_freq", type=int, default=1000, help="display progress every progress_freq steps")
    # parser.add_argument("--save_freq", type=int, default=50000, help="save frequence of model, 0 to disable")

    parser.add_argument("--aggregate_nccl", type=int, default=0, help="whether to use nccl or cpu for gradient aggregation in multi-gpu training")
    parser.add_argument("--gpu_mem_frac", type=float, default=0.9, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int)

    parser.add_argument("--resnet_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000000)

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.output_dir is None:
        args.output_dir = './logs-shape_embedding/'

    if args.resume:
        if args.checkpoint:
            raise ValueError('resume and checkpoint cannot both be specified')
        args.checkpoint = args.output_dir

    # Set dataset & model hparam
    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.dataset_hparams_dict:
        with open(args.dataset_hparams_dict) as f:
            dataset_hparams_dict.update(json.loads(f.read()))
    if args.model_hparams_dict:
        with open(args.model_hparams_dict) as f:
            model_hparams_dict.update(json.loads(f.read()))

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    # Create dataset
    VideoDataset = datasets.get_dataset_class(args.dataset)
    train_dataset = VideoDataset(
        args.input_dir,
        mode='train',
        hparams_dict=dataset_hparams_dict,
        hparams=args.dataset_hparams)
    val_dataset = VideoDataset(
        args.val_input_dir or args.input_dir,
        mode='val',
        hparams_dict=dataset_hparams_dict,
        hparams=args.dataset_hparams)

    batch_size = args.batch_size
    train_tf_dataset = train_dataset.make_dataset(batch_size)
    train_iterator = train_tf_dataset.make_one_shot_iterator()
    train_handle = train_iterator.string_handle()
    val_tf_dataset = val_dataset.make_dataset(batch_size)
    val_iterator = val_tf_dataset.make_one_shot_iterator()
    val_handle = val_iterator.string_handle()
    iterator = tf.data.Iterator.from_string_handle(
        train_handle, train_tf_dataset.output_types, train_tf_dataset.output_shapes)
    # inputs comes from the training dataset by default, unless train_handle is remapped to the val_handles
    inputs = iterator.get_next()
    shape_id_to_embeddings_map = np.load('./metadata/shape_id_to_embeddings_map.npy').item()
    shape_id_to_embeddings_map = tf.convert_to_tensor(np.array(list(shape_id_to_embeddings_map.values())))
    inputs['shape_id_to_embeddings_map'] = shape_id_to_embeddings_map

    # Create model
    variable_scope = tf.get_variable_scope()
    variable_scope.set_use_resource(True)
    num_blocks = (args.resnet_size - 2) // 6
    model = ShapeEmbeddingModel(args.resnet_size, False, 5, 16, 3, 1, None, None, [num_blocks] * 3, [1, 2, 2])
    model.build_graph(inputs)

    with tf.name_scope("parameter_count"):
        # exclude trainable variables that are replicas (used in multi-gpu setting)
        trainable_variables = set(tf.trainable_variables()) # & set(model.saveable_variables)
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in trainable_variables])

    saver = tf.train.Saver(var_list=model.saveable_variables, max_to_keep=2)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    global_step = tf.train.get_or_create_global_step()
    max_steps = args.max_steps
    eval_loss = 1e9

    with tf.Session(config=config) as sess:
        def should(step, freq):
            if freq is None:
                return (step + 1) == (max_steps - start_step)
            else:
                return freq and ((step + 1) % freq == 0 or (step + 1) in (0, max_steps - start_step))

        print("parameter_count =", sess.run(parameter_count))

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.restore(sess, args.checkpoint)
        val_handle_eval = sess.run(val_handle)
        sess.graph.finalize()

        start_step = sess.run(global_step)
        # start at one step earlier to log everything without doing any training
        # step is relative to the start_step
        for step in range(-1, max_steps - start_step):
            fetches = {"global_step": global_step}
            if step >= 0:
                fetches["train_op"] = model.train_op
                fetches["loss"] = model.loss

            run_start_time = time.time()
            results = sess.run(fetches)
            run_elapsed_time = time.time() - run_start_time

            if run_elapsed_time > 1.5 and step > 0 and set(fetches.keys()) == {"global_step", "train_op"}:
                print('running train_op took too long (%0.1fs)' % run_elapsed_time)

            if should(step, args.progress_freq) and step >= 0:
                print(step, "Train loss: {}".format(results['loss']))

            if should(step, args.summary_freq) and step >= 0:
                if val_dataset.num_examples_per_epoch() % args.batch_size != 0:
                    raise ValueError('num_examples_per_epoch should be divided by batch_size, {} % {} != 0'.format(val_dataset.num_examples_per_epoch(), args.batch_size))
                num_iters = val_dataset.num_examples_per_epoch() // args.batch_size
                val_fetches = {'loss': model.loss}
                loss = 0.0
                for i in range(num_iters):
                    val_results = sess.run(val_fetches, feed_dict={train_handle: val_handle_eval})
                    loss += val_results['loss']
                loss /= num_iters
                print(step, "Eval loss: {}".format(loss))

                if loss < eval_loss:
                    print("saving model to", args.output_dir)
                    saver.save(sess, os.path.join(args.output_dir, "model"), global_step=global_step)
                    print("done")
                    eval_loss = loss

            # if should(step, args.save_freq):


if __name__ == '__main__':
    main()
