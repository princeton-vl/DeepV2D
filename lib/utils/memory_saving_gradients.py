from toposort import toposort
import contextlib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import time
import sys
sys.setrecursionlimit(50000)
# refers back to current module if we decide to split helpers out
util = sys.modules[__name__]

# getting rid of "WARNING:tensorflow:VARIABLES collection name is deprecated"
setattr(tf.GraphKeys, "VARIABLES", "variables")

# save original gradients since tf.gradient could be monkey-patched to point
# to our version
from tensorflow.python.ops import gradients as tf_gradients_lib
tf_gradients = tf_gradients_lib.gradients

MIN_CHECKPOINT_NODE_SIZE=1024    # use lower value during testing

# specific versions we can use to do process-wide replacement of tf.gradients
def gradients_speed(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, checkpoints='speed', **kwargs)

def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, checkpoints='memory', **kwargs)

def gradients_collection(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, checkpoints='collection', **kwargs)

def gradients(ys, xs, grad_ys=None, checkpoints='collection', **kwargs):
    '''
    Authors: Tim Salimans & Yaroslav Bulatov

    memory efficient gradient implementation inspired by "Training Deep Nets with Sublinear Memory Cost"
    by Chen et al. 2016 (https://arxiv.org/abs/1604.06174)

    ys,xs,grad_ys,kwargs are the arguments to standard tensorflow tf.gradients
    (https://www.tensorflow.org/versions/r0.12/api_docs/python/train.html#gradients)

    'checkpoints' can either be
        - a list consisting of tensors from the forward pass of the neural net
          that we should re-use when calculating the gradients in the backward pass
          all other tensors that do not appear in this list will be re-computed
        - a string specifying how this list should be determined. currently we support
            - 'speed':  checkpoint all outputs of convolutions and matmuls. these ops are usually the most expensive,
                        so checkpointing them maximizes the running speed
                        (this is a good option if nonlinearities, concats, batchnorms, etc are taking up a lot of memory)
            - 'memory': try to minimize the memory usage
                        (currently using a very simple strategy that identifies a number of bottleneck tensors in the graph to checkpoint)
            - 'collection': look for a tensorflow collection named 'checkpoints', which holds the tensors to checkpoint
    '''

    #    print("Calling memsaving gradients with", checkpoints)
    if not isinstance(ys,list):
        ys = [ys]
    if not isinstance(xs,list):
        xs = [xs]

    bwd_ops = ge.get_backward_walk_ops([y.op for y in ys],
                                       inclusive=True)

    debug_print("bwd_ops: %s", bwd_ops)

    # forward ops are all ops that are candidates for recomputation
    fwd_ops = ge.get_forward_walk_ops([x.op for x in xs],
                                      inclusive=True,
                                      within_ops=bwd_ops)
    debug_print("fwd_ops: %s", fwd_ops)

    # exclude ops with no inputs
    fwd_ops = [op for op in fwd_ops if op.inputs]


    # don't recompute xs, remove variables
    xs_ops = _to_ops(xs)
    fwd_ops = [op for op in fwd_ops if not op in xs_ops]
    fwd_ops = [op for op in fwd_ops if not '/assign' in op.name]
    fwd_ops = [op for op in fwd_ops if not '/Assign' in op.name]
    fwd_ops = [op for op in fwd_ops if not '/read' in op.name]
    ts_all = ge.filter_ts(fwd_ops, True) # get the tensors
    ts_all = [t for t in ts_all if '/read' not in t.name]
    ts_all = set(ts_all) - set(xs) - set(ys)


    checkpoints = 'collection'
    # construct list of tensors to checkpoint during forward pass, if not
    # given as input

    stereo_checkpoints = ge.filter_ts_from_regex(fwd_ops, "add")
    motion_checkpoints = ge.filter_ts_from_regex(fwd_ops, "Conv2D")

    my_ckps = []
    for x in motion_checkpoints:
        if ("motion" in x.name) and ("BatchNorm" not in x.name):
            my_ckps.append(x)
    for x in stereo_checkpoints:
        if ("stereo" in x.name) and ("BatchNorm" not in x.name):
            my_ckps.append(x)

    checkpoints = my_ckps
    checkpoints = list(set(checkpoints).intersection(ts_all))


    # at this point automatic selection happened and checkpoints is list of nodes
    assert isinstance(checkpoints, list)

    debug_print("Checkpoint nodes used: %s", checkpoints)
    # better error handling of special cases
    # xs are already handled as checkpoint nodes, so no need to include them
    xs_intersect_checkpoints = set(xs).intersection(set(checkpoints))
    if xs_intersect_checkpoints:
        debug_print("Warning, some input nodes are also checkpoint nodes: %s",
                    xs_intersect_checkpoints)
    ys_intersect_checkpoints = set(ys).intersection(set(checkpoints))
    debug_print("ys: %s, checkpoints: %s, intersect: %s", ys, checkpoints,
                ys_intersect_checkpoints)
    # saving an output node (ys) gives no benefit in memory while creating
    # new edge cases, exclude them
    if ys_intersect_checkpoints:
        debug_print("Warning, some output nodes are also checkpoints nodes: %s",
              format_ops(ys_intersect_checkpoints))

    # remove initial and terminal nodes from checkpoints list if present
    checkpoints = list(set(checkpoints) - set(ys) - set(xs))

    # check that we have some nodes to checkpoint
    if not checkpoints:
        raise Exception('no checkpoints nodes found or given as input! ')

    # disconnect dependencies between checkpointed tensors
    checkpoints_disconnected = {}
    for x in checkpoints:
        if x.op and x.op.name is not None:
            grad_node = tf.stop_gradient(x, name=x.op.name+"_sg")
        else:
            grad_node = tf.stop_gradient(x)
        checkpoints_disconnected[x] = grad_node

    # partial derivatives to the checkpointed tensors and xs
    ops_to_copy = fast_backward_ops(seed_ops=[y.op for y in ys],
                                    stop_at_ts=checkpoints, within_ops=fwd_ops)
    debug_print("Found %s ops to copy within fwd_ops %s, seed %s, stop_at %s",
                    len(ops_to_copy), fwd_ops, [r.op for r in ys], checkpoints)
    debug_print("ops_to_copy = %s", ops_to_copy)
    debug_print("Processing list %s", ys)
    copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
    copied_ops = info._transformed_ops.values()
    debug_print("Copied %s to %s", ops_to_copy, copied_ops)
    ge.reroute_ts(checkpoints_disconnected.values(), checkpoints_disconnected.keys(), can_modify=copied_ops)
    debug_print("Rewired %s in place of %s restricted to %s",
                checkpoints_disconnected.values(), checkpoints_disconnected.keys(), copied_ops)

    # get gradients with respect to current boundary + original x's
    copied_ys = [info._transformed_ops[y.op]._outputs[0] for y in ys]
    boundary = list(checkpoints_disconnected.values())
    dv = tf_gradients(ys=copied_ys, xs=boundary+xs, grad_ys=grad_ys, **kwargs)
    debug_print("Got gradients %s", dv)
    debug_print("for %s", copied_ys)
    debug_print("with respect to %s", boundary+xs)

    inputs_to_do_before = [y.op for y in ys]
    if grad_ys is not None:
        inputs_to_do_before += grad_ys
    wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
    my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

    # partial derivatives to the checkpointed nodes
    # dictionary of "node: backprop" for nodes in the boundary
    d_checkpoints = {r: dr for r,dr in zip(checkpoints_disconnected.keys(),
                                        dv[:len(checkpoints_disconnected)])}
    # partial derivatives to xs (usually the params of the neural net)
    d_xs = dv[len(checkpoints_disconnected):]

    # incorporate derivatives flowing through the checkpointed nodes
    checkpoints_sorted_lists = tf_toposort(checkpoints, within_ops=fwd_ops)
    for ts in checkpoints_sorted_lists[::-1]:
        debug_print("Processing list %s", ts)
        checkpoints_other = [r for r in checkpoints if r not in ts]
        checkpoints_disconnected_other = [checkpoints_disconnected[r] for r in checkpoints_other]

        # copy part of the graph below current checkpoint node, stopping at
        # other checkpoints nodes
        ops_to_copy = fast_backward_ops(within_ops=fwd_ops, seed_ops=[r.op for r in ts], stop_at_ts=checkpoints_other)
        debug_print("Found %s ops to copy within %s, seed %s, stop_at %s",
                    len(ops_to_copy), fwd_ops, [r.op for r in ts],
                    checkpoints_other)
        debug_print("ops_to_copy = %s", ops_to_copy)
        if not ops_to_copy: # we're done!
            break
        copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
        copied_ops = info._transformed_ops.values()
        debug_print("Copied %s to %s", ops_to_copy, copied_ops)
        ge.reroute_ts(checkpoints_disconnected_other, checkpoints_other, can_modify=copied_ops)
        debug_print("Rewired %s in place of %s restricted to %s",
                    checkpoints_disconnected_other, checkpoints_other, copied_ops)

        # gradient flowing through the checkpointed node
        boundary = [info._transformed_ops[r.op]._outputs[0] for r in ts]
        substitute_backprops = [d_checkpoints[r] for r in ts]
        dv = tf_gradients(boundary,
                          checkpoints_disconnected_other+xs,
                          grad_ys=substitute_backprops, **kwargs)
        debug_print("Got gradients %s", dv)
        debug_print("for %s", boundary)
        debug_print("with respect to %s", checkpoints_disconnected_other+xs)
        debug_print("with boundary backprop substitutions %s", substitute_backprops)


        inputs_to_do_before = [d_checkpoints[r].op for r in ts]
        wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
        my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

        # partial derivatives to the checkpointed nodes
        for r, dr in zip(checkpoints_other, dv[:len(checkpoints_other)]):
            if dr is not None:
                if d_checkpoints[r] is None:
                    d_checkpoints[r] = dr
                else:
                    d_checkpoints[r] += dr

        # partial derivatives to xs (usually the params of the neural net)
        d_xs_new = dv[len(checkpoints_other):]
        for j in range(len(xs)):
            if d_xs_new[j] is not None:
                if d_xs[j] is None:
                    d_xs[j] = d_xs_new[j]
                else:
                    d_xs[j] += d_xs_new[j]

    return d_xs

def tf_toposort(ts, within_ops=None):
    all_ops = ge.get_forward_walk_ops([x.op for x in ts], within_ops=within_ops)

    deps = {}
    for op in all_ops:
        for o in op.outputs:
            deps[o] = set(op.inputs)
    sorted_ts = toposort(deps)

    # only keep the tensors from our original list
    ts_sorted_lists = []
    for l in sorted_ts:
        keep = list(set(l).intersection(ts))
        if keep:
            ts_sorted_lists.append(keep)

    return ts_sorted_lists

def fast_backward_ops(within_ops, seed_ops, stop_at_ts):
    bwd_ops = set(ge.get_backward_walk_ops(seed_ops, stop_at_ts=stop_at_ts))
    ops = bwd_ops.intersection(within_ops).difference([t.op for t in stop_at_ts])
    return list(ops)

@contextlib.contextmanager
def capture_ops():
  """Decorator to capture ops created in the block.
  with capture_ops() as ops:
    # create some ops
  print(ops) # => prints ops created.
  """

  micros = int(time.time()*10**6)
  scope_name = str(micros)
  op_list = []
  with tf.name_scope(scope_name):
    yield op_list

  g = tf.get_default_graph()
  op_list.extend(ge.select_ops(scope_name+"/.*", graph=g))

def _to_op(tensor_or_op):
  if hasattr(tensor_or_op, "op"):
    return tensor_or_op.op
  return tensor_or_op

def _to_ops(iterable):
  if not _is_iterable(iterable):
    return iterable
  return [_to_op(i) for i in iterable]

def _is_iterable(o):
  try:
    _ = iter(o)
  except Exception:
    return False
  return True

DEBUG_LOGGING=False
def debug_print(s, *args):
  """Like logger.log, but also replaces all TensorFlow ops/tensors with their
  names. Sensitive to value of DEBUG_LOGGING, see enable_debug/disable_debug

  Usage:
    debug_print("see tensors %s for %s", tensorlist, [1,2,3])
  """

  if DEBUG_LOGGING:
    formatted_args = [format_ops(arg) for arg in args]
    print("DEBUG "+s % tuple(formatted_args))

def format_ops(ops, sort_outputs=True):
  """Helper method for printing ops. Converts Tensor/Operation op to op.name,
  rest to str(op)."""

  if hasattr(ops, '__iter__') and not isinstance(ops, str):
    l = [(op.name if hasattr(op, "name") else str(op)) for op in ops]
    if sort_outputs:
      return sorted(l)
    return l
  else:
    return ops.name if hasattr(ops, "name") else str(ops)

def my_add_control_inputs(wait_to_do_ops, inputs_to_do_before):
    for op in wait_to_do_ops:
        ci = [i for i in inputs_to_do_before if op.control_inputs is None or i not in op.control_inputs]
        ge.add_control_inputs(op, ci)
