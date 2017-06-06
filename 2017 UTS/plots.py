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
from glob import glob

try:
    from tools.helpers import Progress
except ImportError:
    Progress = lambda iterator: iterator


@click.group()
def main():
    pass

###############################################################################
#                              Simulation Plots                               #
###############################################################################

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
            df = pd.concat([df_old, df], verify_integrity=True,
                           ignore_index=True)
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


###############################################################################
#                             Experimental Plots                              #
###############################################################################
def get_intensities(df, key, timesteps=None):
    # NOT NORMALIZED!
    deteff = df['RAWCOUNTS'][key].attrs['DETEFF']

    intensities = {}
    for key, counts in df['RAWCOUNTS'][key].items():
        counts = counts.value * deteff
        intensities[key] = 1.0 * np.mean(counts[:timesteps], axis=0)  # take time-average
    # we normalize them globally for now, otherwise the solver might have trouble
    normalization = max(sum(x) for x in intensities.values())
    return {key: x / normalization for key, x in intensities.items()}


def recover(df, key, optim_func=plon.lr_recover_l2, mmax=-1, timesteps=None):
    pvec_keys = list(df['INVECS'][key].keys())[:mmax]
    # note that intensities are not normalized!
    intesity_dict = get_intensities(df, key, timesteps=timesteps)
    valid_keys = set(pvec_keys).intersection(set(intesity_dict.keys()))

    pvecs = np.asarray([df['INVECS'][key][pkey].value for pkey in valid_keys])
    intensities = np.asarray([intesity_dict[pkey] for pkey in valid_keys])
    recov, errs = plon.recover(pvecs, intensities, optim_func=optim_func, reterr=True)
    # but also force our normalization on the reconstruction
    recov /= np.sqrt(np.max(np.sum(np.abs(recov)**2, axis=1)))
    return recov, errs


def get_tmat_singles(df):
    DETECTORS = df.attrs['DETECTORS']
    # Average total photon count (for normalization purposes)
    tmat_single = np.array([df['SINGLE_COUNTS'][str(i)] for i in range(len(DETECTORS))], dtype=float)
    deteff = df['SINGLE_COUNTS'].attrs['DETEFF']
    tmat_single = tmat_single * deteff
    # axis = 0 since we flip the tmat later
    tmat_single /= np.max(np.sum(tmat_single, axis=0))
    tmat_single = np.sqrt(tmat_single.T)
    return tmat_single


def get_dip_reference(df):
    try:
        dip_reconstructed = df['DIP_RECONSTRUCTED'].value
        normalization_sq = np.max(np.sum(np.abs(dip_reconstructed)**2, axis=1))
        return dip_reconstructed / np.sqrt(normalization_sq), True
    except KeyError:
        return get_tmat_singles(df), False


def square_mean(x):
    return np.sqrt(np.sum(x**2)) / np.size(x)


EXDESC = {'Fou': 'Fourier', 'Id': 'Identity', 'Swap': 'Swap',
          '_01': 'random', '_02': 'random', '_03': 'random'}
SIZEDESC = {'M2': '2x2', 'M3': '3x3', 'M5': '5x5'}
EXDATAFILES = ['M2Fou.h5', 'M2Id.h5', 'M2_01.h5', 'M2_02.h5', 'M2_03.h5',
               'M3Fou.h5', 'M3Id.h5', 'M3_01.h5', 'M3_02.h5', 'M3_03.h5',
               'M5Fou_rotated.h5', 'M5Id_rotated.h5', 'M5Swap_rotated.h5',
               'M5_01_rotated.h5', 'M5_02_rotated.h5', 'M5_03_rotated.h5']


def id_to_label(the_id):
    label = SIZEDESC[the_id[:2]] + ' '
    for key, value in EXDESC.items():
        if the_id[2:].startswith(key):
            return label + value
    raise ValueError('No label found for', the_id)


def make_explot(datadir, key, ax, dipsref, offset=0.0, color='red'):
    for counter, datafile in enumerate(Progress(EXDATAFILES)):
        with h5py.File(datadir + datafile) as df:
            try:
                recov, _ = recover(df, key)
            except KeyError as e:
                if key == 'RECR_OFFSET':
                    print(key, 'not found for', datafile, '(Using RECR instead.)')
                    recov, _ = recover(df, 'RECR')
                else:
                    raise e

            ref, with_phases = get_dip_reference(df) if dipsref \
                else (df['TARGET'], True)

            if with_phases:
                recov, _ = plon.best_tmat_phases(ref, recov)
                errors = np.abs(recov - ref)
            else:
                errors = np.abs(np.abs(recov) - np.abs(ref))

            artist = ax.scatter([counter + offset], np.mean(errors), marker='D',
                                s=50, c=color)
            ax.scatter([counter + offset] * errors.size, np.ravel(errors), marker='x',
                       s=10, c=color)

    return artist, counter


@main.command()
@click.argument('key')
@click.option('--datadir', help='Directory containing datafiles',
              default='/Users/dsuess/Code/pypllon/data/')
@click.option('--dipsref/--targetref', help='Use dips as reference, otherwise use target',
              default=True)
def explot(key, datadir, dipsref):
    fig = pl.figure(0, figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)

    if key == 'all':
        artist_gauss, counter = make_explot(datadir, 'GAUSS', ax, dipsref,
                                            offset=-.2, color='red')
        artist_recr, counter = make_explot(datadir, 'RECR_OFFSET', ax, dipsref,
                                            offset=.2, color='blue')
        fig.legend([artist_gauss, artist_recr], ['Gaussian', 'RECR'])

    else:
        _, counter = make_explot(datadir, key, ax, dipsref)

    if dipsref:
        ax.set_ylabel(r'$\vert M_\mathrm{dips} - M^\sharp \vert$')
    else:
        ax.set_ylabel(r'$\vert M_\mathrm{target} - M^\sharp \vert$')

    ax.set_xlim(-.5, counter + .5)
    ax.set_xticks(range(counter + 1))
    ax.set_xticklabels([id_to_label(the_id) for the_id in EXDATAFILES],
                       rotation=80)
    ax.grid(True)
    fig.subplots_adjust(bottom=0.2)

    pl.savefig('ex_{}.pdf'.format(key))



if __name__ == '__main__':
    main()
