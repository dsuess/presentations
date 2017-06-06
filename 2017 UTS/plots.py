#!/usr/bin/env python3
# encoding: utf-8

import warnings

import click

import h5py
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pypllon as plon
from pypllon.parsers import load_simdata


@click.group()
def main():
    pass

@main.command()
@click.argument('h5file')
@click.argument('key')
@click.option('--outfile', help='File to save to', default='simdata.h5')
@click.option('--append', help='If we should try to load existing dataframe under key',
              default=True)
def pandarize(h5file, key, outfile, append):
    with h5py.File(h5file, 'r') as infile:
        df = load_simdata(infile)

    print('Number of failed reconstructions:', len(df[df.isnull().any(axis=1)]))

    if append:
        try:
            print('Appending to {}/{}'.format(outfile, key))
            df_old = pd.read_hdf(outfile, key)
            df = pd.concat([df_old, df], verify_integrity=True)
        except (FileNotFoundError, KeyError):
            print('Cannot append to non-existing entry')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df.to_hdf(outfile, key=key, mode='a')
    return 0


@np.vectorize
def recons_error(target, recov):
    a, b = plon.fix_phases('rows_by_max', target, recov)
    return np.linalg.norm(a - b)


@main.command()
@click.argument('key')
@click.option('--infile', help='File to load pandas dataframe from', default='simdata.h5')
@click.option('--errthresh', help='Treshold for succesful reconstruction', default=1e-3)
def simplot(key, infile, errthresh):
    fig = pl.figure(0, figsize=(5, 4))
    print('Loading {} from {}'.format(key, infile))
    df = pd.read_hdf(infile, key)

    # Preprocessing
    full_len = len(df)
    df = df[df['recons'].notnull()]
    print('Dropping {} failed simulations'.format(full_len - len(df)))

    # Creating error & success variables
    df['recons_err'] = recons_error(df['target'], df['recons'])
    df['recons_err'].fillna(1.0, inplace=True)
    df['recons_success'] = df['recons_err'] < errthresh * df['dim']

    # create success matrix
    p_success = df.groupby(['dim', 'measurements']) \
        .recons_success.mean().reset_index().pivot('measurements', 'dim')
    # fill in the gaps using recover probability from lower nr. of measurements
    p_success.fillna(method='ffill', inplace=True)
    # for too low nr. of measurements just set to 0
    p_success.fillna(value=0, inplace=True)
    p_success = p_success[:70]

    fig = pl.figure(0, figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(r'$\mathbb{P}\left(\Vert M - M^\sharp \Vert_2 < 10^{-3} n\right)$')
    ax.set_xlabel(r'$n$ (dimension)')
    ax.set_ylabel(r'$m$ (number of measurements)')

    dims = p_success.columns.levels[1].values
    ax.set_xticks(range(0, max(dims)))
    ax.set_xticklabels(np.sort(dims))
    measurements = p_success.index.values
    yticks = list(range(0, max(measurements), 10))
    ax.set_yticks(yticks - min(measurements))
    ax.set_yticklabels(yticks[::-1])
    ax.grid(False)

    ax.imshow(p_success.values[::-1], aspect='auto', cmap=pl.cm.gray)

    pl.savefig('sim_{}.pdf'.format(key))

if __name__ == '__main__':
    main()
