# coding: utf8
from __future__ import unicode_literals, print_function

import spacy
from prodigy.components.db import connect
from prodigy.components.filters import filter_duplicates
from prodigy.components.loaders import JSONL
from prodigy.core import recipe
from prodigy.models.textcat import TextClassifier

DB = connect()

labels = ['example']
dataset = 'example'
example_jsonl = './data.jsonl'
spacy_model = 'en_core_web_sm'

with open('recipes/custom_template.html') as txt:
    template_text = txt.read()


@recipe('custom_template_recipe_config')
def custom_with_recipe_html_template():
    nlp = spacy.load(spacy_model)
    model = TextClassifier(nlp, labels, long_text=False)
    stream = JSONL(example_jsonl)
    stream = filter_duplicates(stream, by_input=True, by_task=False)

    return {
        'view_id': 'html',
        'dataset': dataset,
        'stream': stream,
        'exclude': [dataset],
        'update': model.update,
        'config': {
            'labels': labels,
            'html_template': template_text
        }
    }


@recipe('custom_template_html_items')
def custom_with_html_on_examples():
    """When specifying html property, template markup is not expanded"""
    nlp = spacy.load(spacy_model)
    model = TextClassifier(nlp, labels, long_text=False)

    def add_html(input_stream):
        for i in input_stream:
            i['html'] = template_text
            yield i

    stream = JSONL(example_jsonl)
    stream = filter_duplicates(stream, by_input=True, by_task=False)
    stream = add_html(stream)

    return {
        'view_id': 'html',
        'dataset': dataset,
        'stream': stream,
        'exclude': [dataset],
        'update': model.update,
        'config': {
            'labels': labels
        }
    }
