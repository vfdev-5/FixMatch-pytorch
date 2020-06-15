
import numpy as np
import torch

from utils import interleave, deinterleave


def np_interleave(x, batch, inverse=False):
    # Code from utils.interleave with tf -> np
    #
    # def interleave(x, batch):
    #     s = x.get_shape().as_list()
    #     return tf.reshape(
    #       tf.transpose(tf.reshape(x, [-1, batch] + s[1:]), [1, 0] + list(range(2, 1+len(s)))), [-1] + s[1:]
    #     )
    s = list(x.shape)
    axes = [-1, batch] if not inverse else [batch, -1]
    return np.reshape(
        np.transpose(np.reshape(x, axes + s[1:]), [1, 0] + list(range(2, 1 + len(s)))),
        [-1] + s[1:],
    )


def np_deinterleave(x, batch):
    return np_interleave(x, batch, inverse=True)


def test_interleave_simple():
    x = torch.arange(960)[:, None]
    y = interleave(x, 15)
    true_y = np_interleave(x.numpy(), 15)
    assert (y.numpy() == true_y).all()


def test_deinterleave_simple():
    x = torch.arange(960)[:, None]
    y = deinterleave(x, 15)
    true_y = np_deinterleave(x.numpy(), 15)
    assert (y.numpy() == true_y).all()


def test_interleave():
    x = torch.rand(960, 3, 32, 32)
    y = interleave(x, 15)
    true_y = np_interleave(x.numpy(), 15)
    assert (y.numpy() == true_y).all()


def test_deinterleave():
    x = torch.rand(960, 3, 32, 32)
    y = deinterleave(x, 15)
    true_y = np_deinterleave(x.numpy(), 15)
    assert (y.numpy() == true_y).all()


def test_interleave_deinterleave():
    x = torch.rand(960, 3, 32, 32)
    y = interleave(x, 15)
    z = deinterleave(y, 15)
    assert (z == x).all()
