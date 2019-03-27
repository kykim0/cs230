"""Sequence-to-sequence RNNs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

from . import constants
from . import model_utils
from tensor2tensor.utils import bleu_hook


class Model(object):
  """RNN based seq2seq model."""

  def __init__(self, mode, hparams):
    """Constructs a model instance."""
    self.mode = mode
    self.hparams = hparams
    self.debug_print = hparams.debug_print

  def _is_train_or_eval(self):
    """Returns true if train or eval mode."""
    return (self.mode == tf.estimator.ModeKeys.TRAIN or
            self.mode == tf.estimator.ModeKeys.EVAL)

  def _make_embeddings(self):
    """Creates embeddings for the model."""
    emb_size = self.hparams.embedding_size

    # Load input vocab file.
    src_vocab_file = os.path.abspath(self.hparams.src_vocab_file)
    with tf.gfile.Open(src_vocab_file) as f:
      self.input_vocab_list = [line.strip().split("\t")[0] for line in f]
      self.input_vocab_size = (
          len(self.input_vocab_list) + self.hparams.num_oov_buckets)

    # Load output vocab file.
    tgt_vocab_file = os.path.abspath(self.hparams.tgt_vocab_file)
    with tf.gfile.Open(tgt_vocab_file) as f:
      self.output_vocab_list = [line.strip().split("\t")[0] for line in f]
      self.output_vocab_size = (
          len(self.output_vocab_list) + self.hparams.num_oov_buckets)

    encoder_embedding = tf.get_variable(
        "encoder_embedding",
        [self.input_vocab_size, emb_size],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=1e-4))

    decoder_embedding = tf.get_variable(
        "decoder_embedding",
        [self.output_vocab_size, emb_size],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=1e-4))

    return encoder_embedding, decoder_embedding

  def _create_cell(self, cell_type, num_units):
    """Creates an RNN cell."""
    if cell_type == "LSTM":
      return tf.contrib.rnn.LSTMCell(num_units)
    if cell_type == "LayerNormBasicLSTM":
      return tf.contrib.rnn.LayerNormBasicLSTMCell(num_units)
    if cell_type == "GRU":
      return tf.contrib.rnn.GRUCell(num_units)
    raise ValueError("Unknown cell type: {}".format(cell_type))

  def _encode(self, encoder_input, encoder_embedding):
    """Builds an encoder from a sequence."""
    if self.debug_print:
      encoder_input = tf.Print(
          encoder_input, [encoder_input], "Encoder input: ", summarize=100)
    encoder_emb_input = tf.nn.embedding_lookup(encoder_embedding, encoder_input)
    cell = self._create_cell(
        self.hparams.encoder_cell_type, self.hparams.encoder_cell_size)

    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        cell, encoder_emb_input, dtype=tf.float32)

    # TODO: Support bidirectional encoding.
    return encoder_output, encoder_state

  def _decode(self, initial_state, decoder_input, decoder_embedding,
              target_sequence_length, memory=None, free_run_ratio=None):
    """Builds a decoder.

    Args:
      initial_state: Initial state for the decoder.
      decoder_input: Input to the decoder.
      decoder_embeeding: Embedding layer for the output.
      target_sequence_length: Length of the target sequence.
      memory: Used for attention.
      free_run_ratio: Proportion of the sequence for which to run in free mode.

    Returns:
      A tuple of decode output and final state.
    """
    decoder_emb_input = tf.nn.embedding_lookup(decoder_embedding, decoder_input)
    if self.debug_print:
      decoder_emb_input = tf.Print(
          decoder_emb_input, [decoder_emb_input], "Decoder embedding input: ",
          summarize=100)

    # Create a decoder cell.
    cell = self._create_cell(
        self.hparams.decoder_cell_type, self.hparams.decoder_cell_size)
    if self.hparams.use_attention:
      attention = tf.contrib.seq2seq.LuongAttention(cell.output_size, memory)
      cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention)
    if self.mode == tf.estimator.ModeKeys.TRAIN:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell,
          input_keep_prob=self.hparams.input_keep_prob,
          output_keep_prob=self.hparams.output_keep_prob,
          state_keep_prob=self.hparams.state_keep_prob)

    helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_emb_input, target_sequence_length)
    if self.mode == tf.estimator.ModeKeys.PREDICT:
      helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
          embedding=decoder_embedding,
          start_tokens=tf.fill([self.hparams.batch_size], 1),
          end_token=2)
    else:
      if free_run_ratio is not None:
        helper = model_utils.MixerTrainingHelper(
            decoder_emb_input, target_sequence_length, decoder_embedding,
            free_run_ratio)
      elif self.hparams.scheduled_sampling_prob > 0.0:
        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            decoder_emb_input, target_sequence_length, decoder_embedding,
            self.hparams.scheduled_sampling_prob)

    if initial_state is None:
      initial_state = cell.zero_state(
          dtype=tf.float32, batch_size=self.hparams.batch_size)
    output_layer = tf.layers.Dense(self.output_vocab_size, name="output_proj")
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell, helper, initial_state, output_layer)

    decoder_output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    return decoder_output, final_state

  def _compute_loss(self, decoder_output, labels, target_sequence_length):
    """Builds a loss op."""
    logits = decoder_output.rnn_output
    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    max_time = tf.shape(labels)[-1]
    mask = tf.sequence_mask(target_sequence_length, dtype=tf.float32)
    batch_size = tf.cast(self.hparams.batch_size, tf.float32)
    loss = tf.reduce_sum(cross_ent * mask) / batch_size
    return loss

  def _compute_mixer_loss(self, decoder_output, labels, target_sequence_length,
                         free_run_ratio):
    """Builds a MIXER loss op."""
    logits = decoder_output.rnn_output
    mask = tf.sequence_mask(target_sequence_length, dtype=tf.float32)
    max_len = tf.shape(mask)[-1]

    mixer_start_steps = model_utils.mixer_start_step(
        free_run_ratio, target_sequence_length)

    mask_mle = tf.sequence_mask(mixer_start_steps, max_len, dtype=tf.float32)
    loss_mle = tf.contrib.seq2seq.sequence_loss(logits, labels, mask_mle)

    sample_ids = decoder_output.sample_id
    argmax_logits = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    mask_rl = (1.0 - mask_mle) * mask

    # Replace the -1s used to indicate where we did not sample.
    policy_ids = (tf.cast(mask_mle, tf.int32) * argmax_logits +
                  tf.cast(mask_rl, tf.int32) * sample_ids)
    policy_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=policy_ids, logits=logits) * tf.cast(mask, tf.float32)

    score = tf.py_func(bleu_hook.compute_bleu, (labels, policy_ids), tf.float32)
    score = tf.expand_dims(score, axis=-1)
    score = score * mask_rl

    loss_rl = -tf.reduce_mean(tf.reduce_sum(score * policy_log_probs, axis=1))

    # Compute the weight to use to combine the two losses.
    anneal_start_step = self.hparams.rl_start_step
    anneal_steps = self.hparams.rl_anneal_steps
    anneal_weight = tf.cast(
        model_utils.anneal_weight(anneal_start_step, anneal_steps), tf.float32)
    tf.summary.scalar("anneal_weight", anneal_weight)

    loss = (1.0 - anneal_weight) * loss_mle + anneal_weight * loss_rl
    return loss

  def _get_optimizer(self):
    """Creates an optimizer."""
    optimizer_type = self.hparams.optimizer_type
    if optimizer_type == "Adam":
      return tf.train.AdamOptimizer(self.hparams.learning_rate)
    # TODO: Support other optimizers.
    raise ValueError("Unsupported optimizer: {}".format(hparam.optimizer_type))

  def model_fn(self, features, labels, mode, params):
    """Builds the model graph."""
    # Create embeddings.
    encoder_emb, decoder_emb = self._make_embeddings()

    # Encoder.
    encoder_input = features[constants.ENCODER_INPUT]
    encoder_output, encoder_state = self._encode(encoder_input, encoder_emb)

    # Decoder.
    decoder_input = features[constants.DECODER_INPUT]
    target_sequence_length = features[constants.TARGET_LENGTH]
    memory = encoder_output if self.hparams.use_attention else None
    free_run_ratio = None
    if self._is_train_or_eval() and self.hparams.use_mixer:
      anneal_start_step = self.hparams.rl_start_step
      anneal_steps = self.hparams.rl_anneal_steps
      free_run_ratio = model_utils.anneal_weight(
          anneal_start_step, anneal_steps)
      tf.summary.scalar("free_run_ratio", free_run_ratio)
    initial_state = None
    if not self.hparams.use_attention:
      initial_state = encoder_state
    decoder_output, final_state = self._decode(
        initial_state, decoder_input, decoder_emb, target_sequence_length,
        memory, free_run_ratio)

    # Loss.
    if self.debug_print:
      labels = tf.Print(labels, [labels], "Labels: ", summarize=100)
    loss, train_op = None, None
    if self._is_train_or_eval():
      if self.hparams.use_mixer:
        loss = self._compute_mixer_loss(
            decoder_output, labels, target_sequence_length, free_run_ratio)
      else:
        loss = self._compute_loss(
            decoder_output, labels, target_sequence_length)

    # Train op.
    if self.mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = self._get_optimizer()
      train_op = tf.contrib.layers.optimize_loss(
          loss=loss,
          global_step=tf.train.get_global_step(),
          learning_rate=self.hparams.learning_rate,
          clip_gradients=self.hparams.max_gradient_norm,
          optimizer=optimizer,
          summaries=["loss"])

    return decoder_output.sample_id, loss, train_op
