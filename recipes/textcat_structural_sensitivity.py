# coding: utf8
from __future__ import unicode_literals, print_function

import cytoolz
import spacy
from prodigy.components import printers
from prodigy.components.db import connect
from prodigy.components.loaders import get_stream
from prodigy.components.sorters import prefer_uncertain, find_with_terms
from prodigy.core import recipe, recipe_args
from prodigy.models.textcat import TextClassifier
from prodigy.util import get_seeds, get_seeds_from_set, log

def structural_sensitivity(nlp, spacy_doc, token_index, label):
    """Determine how sensitive the model is to a token's presence by removing it."""

    base_probability = spacy_doc.cats[label]
    if token_index == 0 or token_index == len(spacy_doc):
        minus_current = spacy_doc[0:max(token_index, 0)].text + spacy_doc[token_index + 1:100000].text
    else:
        minus_current = spacy_doc[0:max(token_index, 0)].text + ' ' + spacy_doc[token_index + 1:100000].text

    if minus_current != '':
        mc_prob = nlp(minus_current).cats[label]
    else:
        # Because the next op is a subtraction of base_probability, this will
        # result in 0 delta for an invalid input.
        mc_prob = base_probability

    mc_delta_prob = base_probability - mc_prob

    return mc_delta_prob, '{0:+6.2f}%'.format(mc_delta_prob * 100)


def attach_structural_sensitivity_data(input_stream, nlp, label):
    """Attach structural sensitivity information to token data with each example"""
    for item in input_stream:
        user_input = item['text']
        doc = nlp(user_input)
        tokens_data = []
        for index, token in enumerate(doc):
            (structure_delta, pretty_delta) = structural_sensitivity(nlp, doc, index, label)
            text_color = 'inherit'
            if structure_delta > 0.1:
                text_color = 'rgba(255,0,0,0.54)'
            elif structure_delta < -0.1:
                text_color = 'rgba(0,128,0,0.87)'
            tokens_data.append({
                't': token.text_with_ws,
                'c': text_color,
                's': max(0.7, 1 + structure_delta * 2),
                'w': structure_delta,
            })
        item['tokens'] = tokens_data
        yield item


# Load custom template
with open('recipes/textcat_structural_sensitivity.html') as txt:
    template_text = txt.read()


@recipe('structcat.teach',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        label=recipe_args['label'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        seeds=recipe_args['seeds'],
        long_text=("Long text", "flag", "L", bool),
        exclude=recipe_args['exclude'])
def teach(dataset, spacy_model, source=None, label='', api=None,
          loader=None, seeds=None, long_text=False, exclude=None):
    """
    Collect the best possible training data for a text classification model
    with the model in the loop. Based on your annotations, Prodigy will decide
    which questions to ask next.
    """
    log('RECIPE: Starting recipe attncat.teach', locals())
    DB = connect()
    nlp = spacy.load(spacy_model)
    log('RECIPE: Creating TextClassifier with model {}'
        .format(spacy_model))
    model = TextClassifier(nlp, label.split(','), long_text=long_text)
    stream = get_stream(source, api, loader, rehash=True, dedup=True,
                        input_key='text')
    if seeds is not None:
        if isinstance(seeds, str) and seeds in DB:
            seeds = get_seeds_from_set(seeds, DB.get_dataset(seeds))
        else:
            seeds = get_seeds(seeds)
        # Find 'seedy' examples
        examples_with_seeds = list(find_with_terms(stream, seeds,
                                                   at_least=10, at_most=1000,
                                                   give_up_after=10000))
        for eg in examples_with_seeds:
            eg.setdefault('meta', {})
            eg['meta']['via_seed'] = True
        print("Found {} examples with seeds".format(len(examples_with_seeds)))
        examples_with_seeds = [task for _, task in model(examples_with_seeds)]
    # Rank the stream. Note this is continuous, as model() is a generator.
    # As we call model.update(), the ranking of examples changes.
    stream = prefer_uncertain(model(stream))
    # Prepend 'seedy' examples, if present
    if seeds:
        log("RECIPE: Prepending examples with seeds to the stream")
        stream = cytoolz.concat((examples_with_seeds, stream))
    stream = attach_structural_sensitivity_data(stream, nlp, label.split(',')[0])
    return {
        'view_id': 'html',
        'dataset': dataset,
        'stream': stream,
        'exclude': exclude,
        'update': model.update,
        'config': {'lang': nlp.lang, 'labels': model.labels, 'html_template': template_text}
    }


@recipe('structcat.eval',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        label=recipe_args['label'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        exclude=recipe_args['exclude'])
def evaluate(dataset, spacy_model, source, label='', api=None,
             loader=None, exclude=None):
    """
    Evaluate a text classification model and build an evaluation set from a
    stream.
    """
    log("RECIPE: Starting recipe attncat.eval", locals())
    nlp = spacy.load(spacy_model, disable=['tagger', 'parser', 'ner'])
    stream = get_stream(source, api, loader)
    stream = attach_structural_sensitivity_data(stream, nlp, label.split(',')[0])
    model = TextClassifier(nlp, label)
    log('RECIPE: Initialised TextClassifier with model {}'
        .format(spacy_model), model.nlp.meta)

    def on_exit(ctrl):
        examples = ctrl.db.get_dataset(dataset)
        data = dict(model.evaluate(examples))
        print(printers.tc_result(data))

    return {
        'view_id': 'html',
        'dataset': dataset,
        'stream': stream,
        'exclude': exclude,
        'on_exit': on_exit,
        'config': {'lang': nlp.lang, 'labels': model.labels, 'html_template': template_text}
    }
