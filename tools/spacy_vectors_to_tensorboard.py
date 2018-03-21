#!/usr/bin/env python
# coding: utf8
"""Visualize spaCy word vectors in Tensorboard.

Adapted from: https://gist.github.com/BrikerMan/7bd4e4bd0a00ac9076986148afc06507
"""
from __future__ import unicode_literals

import numpy as np
import os
import plac
import spacy
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def visualize(model, output_path, tensor_name):
  vectors = model.vocab.vectors
  print('Initializing Tensorboard Vectors: {}'.format(vectors.shape))
  meta_file = "{}.tsv".format(tensor_name)
  placeholder = np.zeros(vectors.shape)

  with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
    for i, (key, vector) in enumerate(vectors.items()):
      placeholder[i] = vector
      text = model.vocab[key].text
      # https://github.com/tensorflow/tensorflow/issues/9094
      file_metadata.write("{0}\n".format(text if text != '' else '<Empty Line>').encode('utf-8'))

  # define the model without training
  sess = tf.InteractiveSession()

  embedding = tf.Variable(placeholder, trainable=False, name=tensor_name)
  tf.global_variables_initializer().run()

  saver = tf.train.Saver()
  writer = tf.summary.FileWriter(output_path, sess.graph)

  # adding into projector
  config = projector.ProjectorConfig()
  embed = config.embeddings.add()
  embed.tensor_name = tensor_name
  embed.metadata_path = meta_file

  # Specify the width and height of a single thumbnail.
  projector.visualize_embeddings(writer, config)
  saver.save(sess, os.path.join(output_path, '{}.ckpt'.format(tensor_name)))


@plac.annotations(
  vectors_loc=("Path to spaCy model that contains word vectors", "positional", None, str),
  out_loc=("Path to output tensorboard vector visualization data", "positional", None, str),
  name=("Human readable name for tsv file and tensor name", "positional", None, str),
)
def main(vectors_loc, out_loc, name="spaCy_vectors"):
  print('Loading spaCy vectors model: {}'.format(vectors_loc))
  nlp = spacy.load(vectors_loc)
  print('Writing Tensorboard visualization: {}'.format(out_loc))
  visualize(nlp, out_loc, name)
  print('Done. Run `tensorboard --logdir={0}` to view in Tensorboard'.format(out_loc))


if __name__ == '__main__':
  plac.call(main)
