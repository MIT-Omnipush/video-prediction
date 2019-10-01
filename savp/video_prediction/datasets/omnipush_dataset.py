import itertools
import os
import re
from collections import OrderedDict

import tensorflow as tf

from video_prediction.utils import tf_utils
from .base_dataset import VarLenFeatureVideoDataset


class OmnipushVideoDataset(VarLenFeatureVideoDataset):
    """
    https://sites.google.com/view/sna-visual-mpc
    """
    def __init__(self, *args, **kwargs):
        super(OmnipushVideoDataset, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = 'images/encoded', (64, 64, 3)
        if self.hparams.use_state:
            self.state_like_names_and_shapes['states'] = 'states', (3,)
            self.state_like_names_and_shapes['shape_ids'] = 'shape_ids', (1,)
            self.action_like_names_and_shapes['actions'] = 'actions', (4,)

    def get_default_hparams_dict(self):
        default_hparams = super(OmnipushVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=12,
            time_shift=1
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parser(self, serialized_example):
        """
        Parses a single tf.train.SequenceExample into images, states, actions, etc tensors.
        """
        features = dict()
        features['sequence_length'] = tf.FixedLenFeature((), tf.int64)
        features['push_name'] = tf.VarLenFeature(tf.string)
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name == 'images':
                features[name] = tf.VarLenFeature(tf.string)
            else:
                features[name] = tf.VarLenFeature(tf.float32)
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            features[name] = tf.VarLenFeature(tf.float32)

        features = tf.parse_single_example(serialized_example, features=features)

        example_sequence_length = features['sequence_length']
        state_like_seqs = OrderedDict()
        action_like_seqs = OrderedDict()
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name == 'images':
                seq = tf.sparse_tensor_to_dense(features[name], '')
            else:
                seq = tf.sparse_tensor_to_dense(features[name])
                seq = tf.reshape(seq, [example_sequence_length] + list(shape))
            state_like_seqs[example_name] = seq
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            seq = tf.sparse_tensor_to_dense(features[name])
            seq = tf.reshape(seq, [example_sequence_length - 1] + list(shape))
            action_like_seqs[example_name] = seq

        state_like_seqs, action_like_seqs = \
            self.slice_sequences(state_like_seqs, action_like_seqs, example_sequence_length)

        # decode and preprocess images on the sampled slice only
        _, image_shape = self.state_like_names_and_shapes['images']
        state_like_seqs['images'] = self.decode_and_preprocess_images(state_like_seqs['images'], image_shape)
        state_like_seqs['push_name'] = tf.sparse_tensor_to_dense(features['push_name'], '')
        return state_like_seqs, action_like_seqs

    @property
    def jpeg_encoding(self):
        return False

    def num_examples_per_epoch(self):
        return self.num_examples
