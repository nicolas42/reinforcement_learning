'''
Helper functions, many from OpenAI baselines
'''
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import collections
import copy 
import os
import multiprocessing
from glob import glob   
import re 
import pandas as pd
from pathlib import Path
home = str(Path.home())

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
          (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
          (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
          (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
          (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def normalise(inputs, inputs_max, inputs_min):
  return (inputs - inputs_min)/(inputs_max - inputs_min)

def denormalise(inputs, inputs_max, inputs_min):
    # x = (max - min) * z + min
    return ((inputs_max - inputs_min)*inputs) + inputs_min

def standardise(inputs, inputs_mean, inputs_std):
    return np.clip((inputs - inputs_mean)/(inputs_std), -5, 5)

def destandardise(inputs, inputs_mean, inputs_std):
    return ((inputs * inputs_std)+ inputs_mean)

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

def load_data_xlxs(DATA_PATH = home + '/data/', sample_name='samples'):
    '''
    Load all the data that is labelled data+num under ~/data/
    '''
    #   folders = glob(DATA_PATH + "*/")
    #   latest = [f.split('/') for f in folders]
    #   latest = [q for l in latest for q in l if q not in home]
    #   l = ''
    #   for item in latest:
    #     l += item
    #   nums = re.findall('(\d+)',l)
    #   nums = np.sort([int(n) for n in nums])
    data = []
    #   for num in nums:
    print("loading data from folder: ", DATA_PATH + '' + sample_name + '.xlsx')

    #   data.extend(pd.read_excel(DATA_PATH + '' + sample_name + '.xlsx', header=None).values())
    data.extend(pd.read_excel(DATA_PATH + '' + sample_name + '.xlsx').as_matrix()[1:,:])
    return np.array(data)

def load_data(DATA_PATH = home + '/data/', sample_name='samples'):
    '''
    Load all the data that is labelled data+num under ~/data/
    '''
    folders = glob(DATA_PATH + "*/")
    latest = [f.split('/') for f in folders]
    latest = [q for l in latest for q in l if q not in home]
    l = ''
    for item in latest:
        l += item
    nums = re.findall('(\d+)',l)
    nums = np.sort([int(n) for n in nums])
    print("loading data from folder: ", nums)
    data = []
    for num in nums:
        data.extend(np.load(DATA_PATH + 'data' + str(num) + '/' + sample_name + '.npy', allow_pickle=True))
    return np.array(data)

def plot_doa(y_data, PATH=None, legend=None, title=None,):
    fig = plt.figure()
    # fig = plt.figure(figsize=(20, 30))
    num_plots = len(y_data)
    cols = [0,4,6,8,12]
    for i,c in zip(range(num_plots), cols):
        plt.plot([_ for _ in range(len(y_data[i]))], y_data[i], c=tableau20[c], alpha=1.0)   
    if legend is not None:
        plt.legend(legend)
    if PATH is not None:
        plt.savefig(PATH + 'torques.png', bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def plot(y_data, PATH=None, legend=None, title=None,):
    fig = plt.figure()
    # fig = plt.figure(figsize=(20, 30))
    num_plots = len(y_data)
    cols = [0,4,6,8,12]
    for i,c in zip(range(num_plots), cols):
        plt.plot([_ for _ in range(len(y_data[i]))], y_data[i], c=tableau20[c], alpha=1.0)   
    if legend is not None:
        plt.legend(legend)
    if PATH is not None:
        plt.savefig(PATH + 'torques.png', bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def subplot2(y_data, PATH=None, legend=None, title=None, x_initial=0, timestep=1/800):
    num_axes = len(y_data)
    fig, axes = plt.subplots(nrows=num_axes, ncols=1)
    # fig.subplots_adjust(hspace=0.5)
    # plt.figure(figsize=(3*num_axes, 5))
    # plt.subplot(221)
    # plt.plot(t, test_prediction[ind1,:,0])
    cols = [0,2,4,6,8,0,2,4,6,8,0,2,4,6,8]
    for j,c in zip(range(num_axes),cols):
        num_plots = len(y_data[j])
        for i in range(num_plots):
            # plt.plot(t, y_valid[ind1,:,0], label = "Label")
            axes[j].plot([(_ + x_initial)*timestep for _ in range(len(y_data[j][i]))], y_data[j][i], c=tableau20[c+i], alpha=1.0) 
        # if legend is not None:
        #   axes[j].legend(legend[j])  
        # if title is not None:
        #   axes[j].set_title(title[j], loc='left')
        #   if 'vel' in title[j]:
        #       axes[j].set_ylabel('rad/s')  
        #   elif 'pos' in title[j]:
        #       axes[j].set_ylabel('rad')  
        #   else:
        #       axes[j].set_ylabel('Nm')  
        #   axes[j].set_xlabel('seconds')  
    if PATH is not None:
        # plt.savefig(PATH + 'torques.png', bbox_inches='tight',dpi=fig.dpi)
        plt.savefig(PATH + 'plot.png')
    else:
        plt.show()

def subplot(y_data, PATH=None, legend=None, title=None, x_initial=0, timestep=1/120, ylim=None):
    num_axes = len(y_data)
    #   fig, axes = plt.subplots(num_axes, figsize=(20*num_axes, 30))
    # fig, axes = plt.subplots(num_axes, figsize=(20, 30))
    fig, axes = plt.subplots(num_axes,  figsize=(10, 3*num_axes))
    cols = [0,2,8,6,4,8,12,14,0,2,6,4,8]
    for j,c in zip(range(num_axes),cols):
        num_plots = len(y_data[j])
        for i in range(num_plots):
        # axes[j].plot([(_ + x_initial)*timestep for _ in range(len(y_data[j][i]))], y_data[j][i], c=tableau20[c+i+3], alpha=1.0) 
        # axes[j].plot([(_ + x_initial)*timestep for _ in range(len(y_data[j][i]))], y_data[j][i], c=tableau20[cols[i]*2], alpha=1.0) 
            axes[j].plot([(_ + x_initial)*timestep for _ in range(len(y_data[j][i]))], y_data[j][i], c=tableau20[cols[i]], alpha=1.0) 
        if legend is not None:
            axes[j].legend(legend[j])  
        if title is not None:
            axes[j].set_title(title[j], loc='left')
            if 'vel' in title[j]:
                axes[j].set_ylabel('rad/s')  
            elif 'pos' in title[j]:
                axes[j].set_ylabel('rad')  
            else:
                axes[j].set_ylabel('Nm')  
            if legend[j] == 'selection':
                axes[j].set_ylim([0,1])

            axes[j].set_xlabel('seconds')  
    if PATH is not None:
        plt.savefig(PATH + 'plot.png', bbox_inches='tight',dpi=fig.dpi)
        plt.close()
    else:
        plt.show()

def print_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

def adjust_shape(placeholder, data):
    '''
    adjust shape of the data to the shape of the placeholder if possible.
    If shape is incompatible, AssertionError is thrown
    Parameters:
        placeholder     tensorflow input placeholder
        data            input data to be (potentially) reshaped to be fed into placeholder
    Returns:
        reshaped data
    '''

    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)

    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]

    assert _check_shape(placeholder_shape, data.shape), \
        'Shape of data {} is not compatible with shape of the placeholder {}'.format(data.shape, placeholder_shape)

    return np.reshape(data, placeholder_shape)


def _check_shape(placeholder_shape, data_shape):
    ''' check if two shapes are compatible (i.e. differ only by dimensions of size 1, or by the batch dimension)'''

    return True
    squeezed_placeholder_shape = _squeeze_shape(placeholder_shape)
    squeezed_data_shape = _squeeze_shape(data_shape)

    for i, s_data in enumerate(squeezed_data_shape):
        s_placeholder = squeezed_placeholder_shape[i]
        if s_placeholder != -1 and s_data != s_placeholder:
            return False

    return True


def _squeeze_shape(shape):
    return [x for x in shape if x != 1]

def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess

def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)

ALREADY_INITIALIZED = set()

def initialize():
    """Initialize all the uninitialized variables in the global scope."""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

def initialize_uninitialized():
    sess = tf.get_default_session()
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
      sess.run(tf.variables_initializer(not_initialized_vars))

def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.
    # Arguments
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.
    """
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)
    return x


def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer


# ================================================================
# Theano-like Function
# ================================================================

def function(inputs, outputs, updates=None, givens=None):
    """Just like Theano function. Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs.
    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).
    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})
        with single_threaded_session():
            initialize()
            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12
    Parameters
    ----------
    inputs: [tf.placeholder, tf.constant, or object with make_feed_dict method]
        list of input arguments
    outputs: [tf.Variable] or tf.Variable
        list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    updates: [tf.Operation] or tf.Operation
        list of update functions or single update function that will be run whenever
        the function is called. The return is ignored.
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (type(inpt) is tf.Tensor and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        self.input_names = {inp.name.split("/")[-1].split(":")[0]: inp for inp in inputs}
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def _feed_input(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = adjust_shape(inpt, value)

    def __call__(self, *args, **kwargs):
        assert len(args) + len(kwargs) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = adjust_shape(inpt, feed_dict.get(inpt, self.givens[inpt]))
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        for inpt_name, value in kwargs.items():
            self._feed_input(feed_dict, self.input_names[inpt_name], value)
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        return results

# ================================================================
# Flat vectors
# ================================================================

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))

def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])

class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        tf.get_default_session().run(self.op, feed_dict={self.theta: theta})

class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return tf.get_default_session().run(self.op)

def flattenallbut0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])

_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)

def get_placeholder(name, dtype, shape):
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        if out.graph == tf.get_default_graph():
            assert dtype1 == dtype and shape1 == shape, \
                'Placeholder with name {} has already been registered and has shape {}, different from requested {}'.format(name, shape1, shape)
            return out

    out = tf.placeholder(dtype=dtype, shape=shape, name=name)
    _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
    return out

def get_placeholder_cached(name):
    return _PLACEHOLDER_CACHE[name][0]

def intprod(x):
    return int(np.prod(x))

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
           summary_tag=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = intprod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = intprod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                            collections=collections)

        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def conv2d_transpose(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
           summary_tag=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        # filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
        filter_shape = [filter_size[0], filter_size[1], num_filters, int(x.get_shape()[-1])]
        # print(tf.shape(x))
        # output_shape = [tf.shape(x)[0], filter_size[0]+1, filter_size[1]+1, num_filters]
        output_shape = [tf.shape(x)[0], filter_size[0]+1, filter_size[1]+1, num_filters]
        # output_shape = tf.shape(x)
        # output_shape = num_filters
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = intprod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = intprod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                            collections=collections)

        # w = [w,h, output, input]
        # output_shape = [N, w+1, h+1, f]

        return tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=stride_shape, padding='SAME') + b
        # return tf.nn.conv2d_transpose(x, w, stride_shape, pad) + b

if __name__=="__main__":
    pass