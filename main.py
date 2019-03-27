"""Main."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import os

import tensorflow as tf

from . import constants
from . import input_fn
from . import model

FLAGS = flags.FLAGS

# Model configurations.
flags.DEFINE_integer("embedding_size", 16, "Embedding size.")

flags.DEFINE_enum("encoder_cell_type", "LSTM",
                  ["LSTM", "LayerNormBasicLSTM", "GRU"],
                  "Type of encoder cell to use.")

flags.DEFINE_integer("encoder_cell_size", 64, "Size of encoder cell.")

flags.DEFINE_enum("decoder_cell_type", "LSTM",
                  ["LSTM", "LayerNormBasicLSTM", "GRU"],
                  "Type of decoder cell to use.")

flags.DEFINE_integer("decoder_cell_size", 64, "Size of decoder cell.")

flags.DEFINE_enum("optimizer_type", "Adam",
                  ["Adam"],
                  "Type of optimizer to use.")

flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")

flags.DEFINE_float("max_gradient_norm", 10.0, "Gradient clipping.")

flags.DEFINE_integer("max_decode_length", 200, "Max length to decode.")

flags.DEFINE_float("scheduled_sampling_prob", 0.0,
                   "Scheduled sampling probabiliy used during trainig.")

flags.DEFINE_bool("use_attention", False, "True to use attention.")

flags.DEFINE_float("input_keep_prob", 1.0, "Dropout input keep probability.")

flags.DEFINE_float("output_keep_prob", 1.0, "Dropout output keep probability.")

flags.DEFINE_float("state_keep_prob", 1.0, "Dropout state keep probability.")

flags.DEFINE_bool("use_mixer", False, "True to use MIXER loss.")

flags.DEFINE_integer("rl_start_step", None, "Step at which to apply MIXER.")

flags.DEFINE_integer("rl_anneal_steps", None, "Number of steps to anneal.")

# Input configurations.
flags.DEFINE_string("src_file", None, "Source input file.")

flags.DEFINE_string("tgt_file", None, "Target input file.")

flags.DEFINE_string("eval_src_file", None, "Eval source input file.")

flags.DEFINE_string("eval_tgt_file", None, "Eval target input file.")

flags.DEFINE_string("src_vocab_file", None, "Source vocab file.")

flags.DEFINE_string("tgt_vocab_file", None, "Target vocab file.")

flags.DEFINE_integer("num_oov_buckets", 5, "Number of OOV buckets.")

flags.DEFINE_integer("buffer_size", 1000, "Shuffle buffer size.")

flags.DEFINE_integer("num_parallel_calls", 12, "Dataset parallelism.")

flags.DEFINE_integer("batch_size", 16, "Batch size.")

# Run configurations.
flags.DEFINE_enum("mode", "train", ["train", "decode"], "Mode to run.")

flags.DEFINE_integer("checkpoints_steps", 300, "Summary steps.")

flags.DEFINE_string("model_dir", None, "Path to model directory.")

flags.DEFINE_integer("summary_steps", 50, "Summary steps.")

flags.DEFINE_integer("train_steps", 100000, "Number of training steps.")

flags.DEFINE_bool("debug_print", False, "True to print debug info.")


def create_hparams():
  """Creates an HParams instance from flags."""
  hparams = tf.contrib.training.HParams()

  # Configs for model.
  hparams.embedding_size = FLAGS.embedding_size
  hparams.encoder_cell_type = FLAGS.encoder_cell_type
  hparams.encoder_cell_size = FLAGS.encoder_cell_size
  hparams.decoder_cell_type = FLAGS.decoder_cell_type
  hparams.decoder_cell_size = FLAGS.decoder_cell_size
  hparams.optimizer_type = FLAGS.optimizer_type
  hparams.learning_rate = FLAGS.learning_rate
  hparams.max_gradient_norm = FLAGS.max_gradient_norm
  hparams.max_decode_length = FLAGS.max_decode_length
  hparams.scheduled_sampling_prob = FLAGS.scheduled_sampling_prob
  hparams.use_attention = FLAGS.use_attention
  hparams.input_keep_prob = FLAGS.input_keep_prob
  hparams.output_keep_prob = FLAGS.output_keep_prob
  hparams.state_keep_prob = FLAGS.state_keep_prob
  hparams.use_mixer = FLAGS.use_mixer
  hparams.rl_start_step = FLAGS.rl_start_step
  hparams.rl_anneal_steps = FLAGS.rl_anneal_steps

  # Configs for input pipeline.
  hparams.src_file = FLAGS.src_file
  hparams.tgt_file = FLAGS.tgt_file
  hparams.eval_src_file = FLAGS.eval_src_file
  hparams.eval_tgt_file = FLAGS.eval_tgt_file
  hparams.src_vocab_file = FLAGS.src_vocab_file
  hparams.tgt_vocab_file = FLAGS.tgt_vocab_file
  hparams.num_oov_buckets = FLAGS.num_oov_buckets
  hparams.buffer_size = FLAGS.buffer_size
  hparams.num_parallel_calls = FLAGS.num_parallel_calls
  hparams.batch_size = FLAGS.batch_size

  # Other configs.
  hparams.debug_print = FLAGS.debug_print

  return hparams


def get_estimator(hparams):
  """Creates an Estimator instance."""
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      save_summary_steps=FLAGS.summary_steps,
      save_checkpoints_steps=FLAGS.checkpoints_steps)

  def _model_fn(features, labels, mode, params):
    seq2seq_model = model.Model(mode, hparams)
    predictions, loss, train_op = seq2seq_model.model_fn(
        features, labels, mode, params)
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)
    return estimator_spec

  estimator = tf.estimator.Estimator(model_fn=_model_fn, config=run_config)
  return estimator


def run_train(hparams):
  """Runs train mode."""

  def _input_fn():
    """Estimator input fn."""
    data_pipeline = input_fn.DataPipeline(hparams)
    return data_pipeline.input_fn(tf.estimator.ModeKeys.TRAIN)

  def _eval_input_fn():
    """Estimator eval input fn."""
    data_pipeline = input_fn.DataPipeline(hparams)
    return data_pipeline.input_fn(tf.estimator.ModeKeys.EVAL)

  train_spec = tf.estimator.TrainSpec(
      input_fn=_input_fn, max_steps=FLAGS.train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=_eval_input_fn)

  estimator = get_estimator(hparams)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def run_decode(hparams):
  """Runs decode mode."""

  def _input_fn():
    """Estimator input fn."""
    data_pipeline = input_fn.DataPipeline(hparams)
    return data_pipeline.input_fn(tf.estimator.ModeKeys.EVAL)

  estimator = get_estimator(hparams)
  predictions = estimator.predict(input_fn=_input_fn)

  with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    for i, pred in enumerate(predictions):
      if i > 10:
        break
      print("pred: {}".format(sess.run(pred)))


def main(unused_argv):
  hparams = create_hparams()
  if FLAGS.mode == "train":
    run_train(hparams)
  elif FLAGS.mode == "decode":
    run_decode(hparams)


if __name__ == "__main__":
  tf.app.run(main=main)
