"""Model utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


def mixer_start_step(free_run_ratio, sequence_length):
  """Steps at which to start sampling."""
  return tf.cast(
      tf.math.round(
          tf.cast(sequence_length, tf.float64) * tf.expand_dims(
              1. - free_run_ratio, -1)), tf.int32)


def anneal_weight(anneal_start_step, anneal_steps):
  """Computes linear anneal weight."""
  global_step = tf.train.get_global_step()
  return tf.clip_by_value(
      (global_step - anneal_start_step) / anneal_steps, 0.0, 1.0)


class MixerTrainingHelper(tf.contrib.seq2seq.TrainingHelper):
  """Training helper for MIXER training.

  See ScheduledEmbeddingTrainingHelper.
  """

  def __init__(self, inputs, sequence_length, embedding, free_run_ratio,
               time_major=False, seed=None, name=None):
    """Constructor for MixerTrainingHelper."""
    with tf.name_scope(name, "MixerTrainingHelper",
                       [embedding, free_run_ratio]):
      self._embedding_fn = lambda ids: tf.nn.embedding_lookup(embedding, ids)
      self._sample_start_step = mixer_start_step(
          free_run_ratio, sequence_length)
      self._sequence_length = sequence_length
      self._seed = seed
      super(MixerTrainingHelper, self).__init__(
          inputs=inputs,
          sequence_length=sequence_length,
          time_major=time_major,
          name=name)

  def sample(self, time, outputs, state, name=None):
    # Return -1s where we did not sample, and sample_ids elsewhere.
    with tf.name_scope(name, "MixerTrainingHelperSample",
                       [time, outputs, state]):
      select_sample = time > self._sample_start_step
      sample_id_sampler = tfp.distributions.Categorical(logits=outputs)
      return tf.where(
          select_sample,
          sample_id_sampler.sample(seed=self._seed),
          tf.fill([self.batch_size], -1))

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    with tf.name_scope(name, "MixerTrainingHelperNextInputs",
                       [time, outputs, state, sample_ids]):
      (finished, base_next_inputs, state) = (
          super(MixerTrainingHelper, self).next_inputs(
              time=time,
              outputs=outputs,
              state=state,
              sample_ids=sample_ids,
              name=name))

      def maybe_sample():
        """Perform sampling."""
        where_sampling = tf.cast(tf.where(sample_ids > -1), tf.int32)
        where_not_sampling = tf.cast(tf.where(sample_ids <= -1), tf.int32)
        sample_ids_sampling = tf.gather_nd(sample_ids, where_sampling)
        inputs_not_sampling = tf.gather_nd(base_next_inputs, where_not_sampling)
        sampled_next_inputs = self._embedding_fn(sample_ids_sampling)
        base_shape = tf.shape(base_next_inputs)
        return (tf.scatter_nd(indices=where_sampling,
                              updates=sampled_next_inputs,
                              shape=base_shape)
                + tf.scatter_nd(indices=where_not_sampling,
                                updates=inputs_not_sampling,
                                shape=base_shape))

      all_finished = tf.reduce_all(finished)
      next_inputs = tf.cond(
          all_finished, lambda: base_next_inputs, maybe_sample)
      return (finished, next_inputs, state)
