"""Get attention weights out of the ParametricAttention layer."""
from contextlib import contextmanager

from thinc.api import layerize


def find_attn_layer(model):
    queue = [model]
    seen = set()
    for layer in queue:
        names = [child.name for child in layer._layers]
        if 'para-attn' in names:
            return layer, names.index('para-attn')
        if id(layer) not in seen:
            queue.extend(layer._layers)
        seen.add(id(layer))
    return None, -1


def create_attn_proxy(attn):
    """Return a proxy to the attention layer which will fetch the attention
    weights on each call, appending them to the list 'output'.
    """
    output = []

    def get_weights(Xs_lengths, drop=0.):
        Xs, lengths = Xs_lengths
        output.append(attn._get_attention(attn.Q, Xs, lengths)[0])
        return attn.begin_update(Xs_lengths, drop=drop)

    return output, layerize(get_weights)


@contextmanager
def get_attention_weights(textcat):
    """Wrap the attention layer of the textcat with a function to
    intercept the attention weights. We replace the attention component
    with our wrapper in the pipeline for the duration of the context manager.
    On exit, we put everything back.
    """
    parent, i = find_attn_layer(textcat.model)
    if parent is not None:
        output_vars, wrapped = create_attn_proxy(parent._layers[i])
        parent._layers[i] = wrapped
        yield output_vars
    else:
        yield None
