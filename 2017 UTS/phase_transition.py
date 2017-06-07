# coding: utf-8
from os import environ
from functools import partial

import cvxpy
import h5py
import ipyparallel
import numpy as np
import pypllon as plon
from pypllon.ccg_haar import unitary_haar
from pypllon.parsers import load_simdata
from scipy.linalg import dft
from tools.helpers import Progress

CLUSTER_ID = environ.get('CLUSTER_ID', None)
_clients = ipyparallel.Client(cluster_id=CLUSTER_ID)
_view = _clients.load_balanced_view()
print("Kernels available: {}".format(len(_clients)))

RGEN = np.random.RandomState(seed=274395387)

###############################################################################
# lower and upper bounds of number of measurements (as a function of size)
LBOUND = lambda dim: 5 * dim - 25
UBOUND = lambda dim: 6 * dim - 5
MSKIP = 2

# Buffer file
OUTFILE = 'noiseless_gaussian.h5'

# the maximal dimension to check
MIN_DIM = 2
MAX_DIM = 12

# Number of random matrices to sample
SAMPLES = 100

# Fraction of measurement vectors sampled from RECR ensemble
# (the rest is sampled from Gaussian ensemble)
INVECS_GENERATOR = plon.invecs_gaussian

# the single error-componenet standard deviation
SIGMA = 0.

# the optimization function to use (from phaselift.routines)
OPTIM_FUNC = partial(plon.lr_recover_l2, solver=cvxpy.MOSEK)

###############################################################################

def generate_tmats(dim, rgen):
    assert SAMPLES >= 3
    mats = {'ID': np.eye(dim), 'SWAP': np.eye(dim)[::-1]}
    mats.update({'DFT': dft(dim, scale='sqrtn')})
    mats.update({'RAND_%i' % i: unitary_haar(dim, rgen=rgen)
                 for i in range(SAMPLES - 3)})
    return mats


def write_h5_header(h5file):
    rstate = RGEN.get_state()

    h5file.attrs['random_gen'] = rstate[0]
    h5file.attrs['random_keys'] = rstate[1]
    h5file.attrs['random_pos'] = rstate[2]
    h5file.attrs['random_has_gauss'] = rstate[3]
    h5file.attrs['random_cached_gaussian'] = rstate[4]
    h5file.attrs['cvxpy_version'] = cvxpy.__version__

    setupvars = ((key, value) for key, value in globals().items()
                 if key.isupper())
    for key, value in setupvars:
        try:
            h5file.attrs[key] = value
        except TypeError:
            pass


def write_h5_invecs(h5group, invecs):
    h5group['INVECS'] = invecs


def write_h5_intensities(h5group, intensities):
    group = h5group.create_group('INTENSITIES')
    for key, val in intensities.items():
        group[str(key)] = val


def write_h5_tmats(h5group, tmats):
    group = h5group.create_group('TARGETS')
    for key, val in tmats.items():
        group[str(key)] = val


def write_h5_result(h5group, nr_measurements, result):
    group = h5group.create_group('RECOV_%i' % nr_measurements)
    group.attrs['NR_MEASUREMENTS'] = nr_measurements
    for name, success, recons, errs in result:
        if not success:
            continue
        group[str(name)] = recons
        group[str(name)].attrs['errs'] = errs


_clients[:].push({'OPTIM_FUNC': OPTIM_FUNC});
with _clients[:].sync_imports():
    import pypllon
    import cvxpy
    import numpy


def recover(args):
    name, invecs, intensities = args
    try:
        recons, err = pypllon.routines.recover(invecs, intensities, optim_func=OPTIM_FUNC,
                                      reterr=True)
        return name, True, recons, err
    except (cvxpy.SolverError, RuntimeError, ZeroDivisionError) as e:
        print("Failed recovering {} with m={}".format(name, len(invecs)))
        print(e)
        dim = invecs.shape[1]
        return name, False, numpy.zeros((dim, dim)), numpy.zeros(dim)


with h5py.File(OUTFILE, 'w') as df:
    write_h5_header(df)
    for dim in range(MIN_DIM, MAX_DIM + 1):
        dimgroup = df.create_group('DIM=%i' % dim)
        dimgroup.attrs['DIM'] = dim

        invecs = INVECS_GENERATOR(dim, UBOUND(dim), rgen=RGEN)
        write_h5_invecs(dimgroup, invecs)

        tmats = generate_tmats(dim, rgen=RGEN)
        write_h5_tmats(dimgroup, tmats)

        intensities = {name: np.abs(invecs @ tmat.T)**2 + SIGMA * RGEN.randn(*invecs.shape)
                       for name, tmat in tmats.items()}
        write_h5_intensities(dimgroup, intensities)

        nr_measurements = np.arange(max(LBOUND(dim), 1), UBOUND(dim) + 1, MSKIP)
        dimgroup.attrs['NR_MEASUREMENTS'] = nr_measurements

        for m in Progress(nr_measurements):
            sequence = [(name, invecs[:m], val[:m]) for name, val in intensities.items()]
            result = _view.map_sync(recover, sequence)
            write_h5_result(dimgroup, m, result)
            df.flush()
        print('Finished dim=', dim)

print('DONE')
