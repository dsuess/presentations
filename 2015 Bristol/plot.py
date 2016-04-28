#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

# import matplotlib as mpl
# mpl.use('Qt4Agg')
import sys
sys.path.append('/Users/dsuess/Code/phaselift/')

import h5py
from matplotlib import pyplot as pl
from matplotlib import ticker

import numpy as np

import postprocessing as pp


def deviation_plot_single(filename, thresh):
    ax = pl.subplot(111)
    data = pp.recons_heatmap(filename, error_thresh=thresh).T
    ax.axis((1.5, data.shape[1] - .5, -.5, data.shape[0] - .5))

    x = np.arange(0, data.shape[1] + 1)
    ax.plot(x, 4*x - 4, color='r', ls='-')

    with h5py.File(filename, 'r') as datafile:
        shape = pp._get_experiment_shape(datafile)

    for dim, vmin, vmax in shape:
        data[:vmin, dim] = 0
        data[vmax + 1:, dim] = 1.

    ax.imshow(data, interpolation='nearest', aspect='auto', vmin=0, vmax=1)

    ax.set_xlabel(r"\# {\lf Modes} $n$")
    ax.set_ylabel(r"\# {\lf Preparations} $m$")

    # cb = pl.colorbar(ax.get_images()[0], ticks=[0, 1], orientation='horizontal',
                     # pad=0.18)
    # cb.ax.tick_params(axis='y', direction='out')

    # ax.set_title(r"$\mathbb{P}\left(\Vert M - M^\sharp\Vert_2 < 10^{-5}\right)$")

    pl.tight_layout()
    return ax

if __name__ == '__main__':
    pl.figure(0, figsize=(2.6, 3))
    ax = deviation_plot_single('noiseless.h5', thresh=1e-5)
    ax.set_title(r"$\mathbb{P}\left(\Vert M - M^\sharp\Vert_2 < 10^{-5}\right)$")
    pl.savefig('noiseless.pdf', bbox_inches='tight')
    pl.close()

    pl.figure(0, figsize=(2.6, 3))
    ax = deviation_plot_single('noisy_l2_sigma=0.001.h5', thresh=1e-2)
    ax.set_title(r"$\mathbb{P}\left(\Vert M - M^\sharp\Vert_2 < 10^{-2}\right)$")
    pl.savefig('noisy.pdf', bbox_inches='tight')
    pl.close()
