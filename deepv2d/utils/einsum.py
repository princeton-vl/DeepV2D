import tensorflow as tf
import numpy as np
import re
import string

def einsum(equation, *inputs):

    equation = equation.replace(' ', '')
    input_shapes = [x.get_shape() for x in list(inputs)]
    match = re.match('^([a-zA-Z,.]+)(->[a-zA-Z.]*)?$', equation)
    if not match:
        raise ValueError('Indices have incorrect format: %s' % equation)

    input_axis_labels = match.group(1).split(',')
    output_axis_labels = match.group(2)[2:] if match.group(2) else None

    if len(input_shapes) != len(input_axis_labels):
        raise ValueError('Got %d arguments for equation "%s", expecting %d' %
                        (len(input_shapes), equation, len(input_axis_labels)))

    # Resolve Ellipsis
    # Assign axes labels for unspecified dimensions in inputs. Labels taken
    # from unused labels. Follow numpy einsum broadcasting conventions for
    # tensors of different length and unlabeled output.
    ellipsis_axes = ''
    if '...' in equation:
        unused = ''.join([c for c in string.ascii_letters
                        if c not in ''.join(input_axis_labels)])
        for i, ax in enumerate(input_axis_labels):
            if '...' in ax:
                parts = ax.split('...')
                if len(parts) != 2:
                    raise ValueError('Unable to resolve ellipsis. Excess number found.')
                if input_shapes[i].ndims is None:
                    raise ValueError('Unable to statically infer ellipsis axes.')
                n = input_shapes[i].ndims - len(''.join(parts))
                if n < 0:
                    raise ValueError('Ellipses lengths do not match.')
                if len(unused) < n:
                    raise ValueError(
                        'Unable to resolve ellipsis, too many distinct labels.')
                replace_axes = unused[-n:] if n > 0 else ''
                input_axis_labels[i] = input_axis_labels[i].replace('...',
                                                                    replace_axes)
                if len(replace_axes) > len(ellipsis_axes):
                    ellipsis_axes = replace_axes

    equation = equation.replace('...', ellipsis_axes)
    out = tf.einsum(equation, *inputs)
    tf.add_to_collection("checkpoints", out)
    return out
