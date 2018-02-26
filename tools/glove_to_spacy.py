#!/usr/bin/env python
# coding: utf8
"""Load vectors for a language trained using GloVe and exported in text file format
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals

import numpy
import plac
import spacy
import tqdm


@plac.annotations(
    vectors_loc=("Path to GloVe pre-trained vectors .txt file", "positional", None, str),
    out_loc=("Path to output model that contains vectors", "positional", None, str),
    lang=("Optional language ID. If not set, 'en' will be used.", "positional", None, str),
)
def main(vectors_loc, out_loc, lang=None):
    if lang is None:
        lang = 'en'
    nlp = spacy.blank(lang)
    print('Loading GloVe vectors: {}'.format(vectors_loc))
    with open(vectors_loc, 'r') as file_:
        lines = file_.readlines()
        print('Assigning {:,} spaCy vectors'.format(len(lines)))
        for line in tqdm.tqdm(lines, leave=False):
            pieces = line.split(' ')
            word = pieces[0]
            vector = numpy.asarray([float(v) for v in pieces[1:]], dtype='f')
            nlp.vocab.set_vector(word, vector)
    print('Saving spaCy vector model: {}'.format(out_loc))
    nlp.to_disk(out_loc)
    print('Done.')


if __name__ == '__main__':
    plac.call(main)
