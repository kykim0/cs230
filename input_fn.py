"""Input pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from . import constants


class DataPipeline(object):
  """Data pipeline with utils constructed using tf.data."""

  def __init__(self, hparams):
    """Constructs an Estimator data pipeline."""
    self.hparams = hparams

    # Create vocab tables.
    src_vocab_file = os.path.abspath(self.hparams.src_vocab_file)
    self.src_vocab_table = tf.contrib.lookup.index_table_from_file(
        src_vocab_file, num_oov_buckets=self.hparams.num_oov_buckets,
        default_value=constants.UNK_ID)
    self.src_str_table = tf.contrib.lookup.index_to_string_table_from_file(
        src_vocab_file, default_value=constants.UNK)
    tgt_vocab_file = os.path.abspath(self.hparams.tgt_vocab_file)
    self.tgt_vocab_table = tf.contrib.lookup.index_table_from_file(
        tgt_vocab_file, num_oov_buckets=self.hparams.num_oov_buckets,
        default_value=constants.UNK_ID)
    self.tgt_str_table = tf.contrib.lookup.index_to_string_table_from_file(
        tgt_vocab_file, default_value=constants.UNK)

  def input_fn(self, mode):
    """Estimator input function."""
    if mode == tf.estimator.ModeKeys.TRAIN:
      src_file = os.path.abspath(self.hparams.src_file)
      tgt_file = os.path.abspath(self.hparams.tgt_file)
    elif (mode == tf.estimator.ModeKeys.EVAL or
          mode == tf.estimator.ModeKeys.PREDICT):
      src_file = os.path.abspath(self.hparams.eval_src_file)
      tgt_file = os.path.abspath(self.hparams.eval_tgt_file)
    else:
      raise ValueError("Unsupported mode.")

    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)

    buffer_size = self.hparams.buffer_size
    num_parallel_calls = self.hparams.num_parallel_calls

    # Combine source and target.
    dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    # Randomly shuffle and sample.
    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.apply(
          tf.data.experimental.shuffle_and_repeat(buffer_size=buffer_size))

    # Split strings.
    dataset = dataset.map(
        lambda src, tgt:(
            tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_parallel_calls)

    # Filter zero length sequences.
    dataset = dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    # Convert strings to ids.
    dataset = dataset.map(
        lambda src, tgt: (tf.cast(self.src_vocab_table.lookup(src), tf.int32),
                          tf.cast(self.tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls)

    # Transform into the format expected by Estimator input function.
    sos = tf.constant(constants.SOS, dtype=tf.string)
    eos = tf.constant(constants.EOS, dtype=tf.string)
    src_eos_id = tf.cast(self.src_vocab_table.lookup(eos), tf.int32)
    tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(sos), tf.int32)
    tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(eos), tf.int32)
    dataset = dataset.map(
        lambda src, tgt: (
            src,
            tf.concat(([tgt_sos_id], tgt), axis=0),
            tf.concat((tgt, [tgt_eos_id]), axis=0)),
        num_parallel_calls=num_parallel_calls)

    # Compute the sequence lengths.
    dataset = dataset.map(
        lambda src, tgt_in, tgt_out: (
            {constants.ENCODER_INPUT: src,
             constants.DECODER_INPUT: tgt_in,
             constants.SOURCE_LENGTH: tf.size(src),
             constants.TARGET_LENGTH: tf.size(tgt_in)},
            tgt_out),
        num_parallel_calls=num_parallel_calls)

    # Create batches of examples.
    padded_shapes = (
        {constants.ENCODER_INPUT: [None],
         constants.DECODER_INPUT: [None],
         constants.SOURCE_LENGTH: [],
         constants.TARGET_LENGTH: []},
        [None])
    padded_values = (
        {constants.ENCODER_INPUT: src_eos_id,
         constants.DECODER_INPUT: tgt_eos_id,
         constants.SOURCE_LENGTH: 0,
         constants.TARGET_LENGTH: 0},
        tgt_eos_id)
    dataset = dataset.padded_batch(
        self.hparams.batch_size,
        padded_shapes=padded_shapes,
        padding_values=padded_values)

    # Prefetch.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
