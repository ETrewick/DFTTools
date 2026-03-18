import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from contextlib import nullcontext
import sympy as sp
from sympy.physics.quantum.cg import CG

np.set_printoptions(precision=3, suppress=True, linewidth=10000000)




def main():
    parser = argparse.ArgumentParser(
        description="Occupation matrix tools"
    )

    parser.add_argument(
        "--quanty",
        action="append",
        metavar="FILE",
        help="Quanty density file(s). Can be repeated."
    )

    parser.add_argument(
        "--qe",
        nargs="+",
        action=QEAction,
        metavar=("FILE", "INT"),
        help="QE density file followed by the dimensions l of each manifild. Use negative l to discard. Can be repeated."
    )
    parser.add_argument(
        "--nspin",
        type=int,
        choices=[2, 4],
        default=2,
        help="Number of spins. Must be 2 or 4."
    )
    parser.add_argument(
        '--savefigs', 
        action='store_true',
        help='Save figures'
    )
    parser.add_argument(
        '--filetype',
        type=str,
        # choices=['pdf', 'png', 'eps'],  # restrict to these types
        default='png',                  # optional default
        help='File type/extension to save the figure '
    )
    parser.add_argument(
        '--hideplot',
        action='store_true',
        default=False,
        help="Don't show the interactive plotting window"
    )

    args = parser.parse_args()


    if not args.quanty and not args.qe:
        parser.error("No input provided. Use --quanty or --qe")

    if args.quanty:
        for fname in args.quanty:
            quanty_4f_dens, quanty_4f_evals, quanty_4f_evecs = load_quanty_density(fname)
            outfname = fname if args.savefigs else None
            plot_eigenvectors(
                quanty_4f_evals,
                quanty_4f_evecs,
                "quanty",
                npts=50,
                fname=outfname,
                ftype=args.filetype
            )

    if args.qe:
        for fname, integers in args.qe:
            qe_dens_list = []
            readocc = {2: readocc_2spin, 4: readocc_4spin}[args.nspin]
            with open(fname, 'rb') as f:
                for l in integers:
                    ndim = np.abs(l)*2+1
                    qe_dens_list.append(readocc(f, ndim))
            for i, (l, dens) in enumerate(zip(integers, qe_dens_list)):
                if l <= 0: # don't plot negative l orbitals (or s orbitals, lol)
                    continue
                # for plotting I have to diagonalise the QE one
                qe_E, qe_R = make_diagonal(dens)
                outfname = f'{fname}_{i+1}L{l}'if args.savefigs else None
                qe_figs = plot_eigenvectors(qe_E, qe_R, 'QE', npts=50, fname=outfname, ftype=args.filetype)
    if not args.hideplot:
        plt.show()


class QEAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) < 2:
            parser.error("--qe requires: FILE INT [INT ...]")

        filename = values[0]

        try:
            integers = [int(v) for v in values[1:]]
        except ValueError:
            parser.error("All arguments after FILE must be integers.")

        # Append to list (since we allow multiple --qe)
        current = getattr(namespace, self.dest, None)
        if current is None:
            current = []
        current.append((filename, integers))

        setattr(namespace, self.dest, current)

    # '''
    #     I'm not even removely consistent with names. The density matrix is sometimes called the occupations matrix (O)
    #     The "natural orbtials" are the matrix which diagonalises the density matrix and is often called eigenvectors (evecs or evecs_right), or R (for rotation matrix)
    #     The the occupations of this matrix are sometimes called occupations, or simply eigenvalues (evals, or E) 

    #     I can only say that I'm sorry, and that in literature it's just as bad (although most papers are somewhat internally consistent)

    #     For testing, this line is helpful:
    # '''

    # # load a starting point from QE (we need the 5d electron density matrix)
    # path_prefix='/home/ludoric/Documents/PhD_stuff/quanty/QE_calc_from_quanty/'
    # qe_occup_fname = path_prefix+'SmN_paw/it2_edited-occup.txt'
    # qe_dens_4f, qe_dens_5d = readocc_4f_5d(qe_occup_fname)
    # # for plotting I have to diagonalise the QE one
    # qe_E, qe_R = make_diagonal(qe_dens_4f)
    # qe_figs = plot_eigenvectors(qe_E, qe_R, 'QE', npts=50, fname=None)
    # 

    # # load the output from quanty
    # quanty_dens_fname = path_prefix+'Dy_1_Density_matrix_full.txt'
    # quanty_4f_dens, quanty_4f_evals, quanty_4f_evecs = load_quanty_density(quanty_dens_fname)
    # qu_figs = plot_eigenvectors(quanty_4f_evals, quanty_4f_evecs, 'quanty', npts=50, fname=None)

    # 
    # # note that the QE and Quanty arrays use different basis functions, so we must perform a rotation between them
    # quanty_4f_dens_in_qe = convertquanty2qe(quanty_4f_dens)
    # # additionally QE wants this to be all real (lose all angular momentum?)
    # quanty_4f_dens_in_qe_real = 0.5 * (quanty_4f_dens_in_qe + quanty_4f_dens_in_qe.conj())

    # # check that the rotation and plotting works correctly
    # qu_E, qu_R = make_diagonal(quanty_4f_dens_in_qe_real)
    # qu_figs2 = plot_eigenvectors(qu_E, qu_R, 'QE', npts=50, fname=None, title='quanty in QE format')

    # # write the bloody thing back out for QE
    # qe_occup_out_fname = path_prefix+'occup.txt-output'
    # writeocc_4f_5d(qe_occup_out_fname, quanty_4f_dens_in_qe_real, qe_dens_5d)
    # writeocc_4spin('occup.txt-output4spin', quanty_4f_dens_in_qe_real)

    # # try not to cry when it doesn't work

    # # qu_figs2 = plot_density(quanty_4f_dens, 'quanty', npts=50, fname=None)

    # plt.show()


def parse_RSPT_matrix(text,l=3):
    '''
          ID Sm4f     Local dmtx: rho
 Real:
   0.02547378  0.00815632 -0.05993573 -0.00257631 -0.00000000  0.00000000  0.00000000 -0.00000000  0.00000000  0.00000000  0.00000000  0.00000000 -0.00000000  0.00000000
   0.00815632  0.03018635 -0.00163622 -0.03762593  0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000  0.00000000 -0.00000000
  -0.05993573 -0.00163622  0.94585574  0.02814897  0.00000000 -0.00000000  0.00000000  0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000
  -0.00257631 -0.03762593  0.02814897  0.63914151 -0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000  0.00000000 -0.00000000  0.00000000
  -0.00000000  0.00000000 -0.00000000 -0.00000000  0.01182704  0.00630999 -0.04423478  0.00225773 -0.00000000 -0.00000000 -0.00000000  0.00000000  0.00000000 -0.00000000
   0.00000000  0.00000000 -0.00000000  0.00000000  0.00630999  0.01392475  0.00369375 -0.05147192 -0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000000
   0.00000000  0.00000000  0.00000000  0.00000000 -0.04423478  0.00369375  0.93363191 -0.04293808  0.00000000 -0.00000000  0.00000000 -0.00000000  0.00000000  0.00000000
  -0.00000000 -0.00000000  0.00000000  0.00000000  0.00225773 -0.05147192 -0.04293808  0.70466390 -0.00000000 -0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000
   0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000  0.02516274  0.00724386 -0.00134695 -0.00000000  0.00000000 -0.00000000
   0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000  0.00724386  0.02617330 -0.05493009  0.00000000 -0.00000000  0.00000000
   0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000000  0.00000000 -0.00134695 -0.05493009  0.72313151 -0.00000000 -0.00000000 -0.00000000
   0.00000000  0.00000000 -0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000  0.03617594 -0.06515291  0.00522198
   0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000  0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000 -0.06515291  0.89297465 -0.08279660
   0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000  0.00522198 -0.08279660  0.51986984
 Imag:
  -0.00000000  0.00000003 -0.00000001 -0.00000003  0.00000000 -0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000  0.00000000  0.00000000
  -0.00000003 -0.00000000  0.00000001  0.00000001 -0.00000000 -0.00000000  0.00000000 -0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000 -0.00000000
   0.00000001 -0.00000001 -0.00000000  0.00000023 -0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000
   0.00000003 -0.00000001 -0.00000023  0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000
  -0.00000000  0.00000000  0.00000000 -0.00000000  0.00000000 -0.00000010  0.00000001  0.00000004  0.00000000  0.00000000  0.00000000  0.00000000 -0.00000000  0.00000000
   0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000010  0.00000000 -0.00000007 -0.00000001  0.00000000  0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000
   0.00000000 -0.00000000  0.00000000  0.00000000 -0.00000001  0.00000007 -0.00000000 -0.00000093 -0.00000000  0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000
  -0.00000000 -0.00000000  0.00000000  0.00000000 -0.00000004  0.00000001  0.00000093 -0.00000000  0.00000000  0.00000000 -0.00000000  0.00000000 -0.00000000  0.00000000
  -0.00000000  0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000004 -0.00000001  0.00000000 -0.00000000  0.00000000
  -0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000  0.00000000 -0.00000004  0.00000000  0.00000000 -0.00000000  0.00000000 -0.00000000
  -0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000  0.00000000  0.00000000  0.00000000  0.00000001 -0.00000000 -0.00000000  0.00000000  0.00000000  0.00000000
   0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000 -0.00000000  0.00000000  0.00000001 -0.00000001
  -0.00000000 -0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000 -0.00000001  0.00000000 -0.00000013
  -0.00000000  0.00000000  0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000 -0.00000000  0.00000000  0.00000000  0.00000001  0.00000013  0.00000000
    '''
    lines = text.strip().splitlines()
    try:
        imag = lines.index(" Imaginary part:")
    except:
        imag = lines.index(" Imag:")
    ndim =  (l*2+1)*2
    start = imag-ndim
    occupations_real = np.array([np.fromstring(lines[start + i], sep=' ') for i in range(ndim)])
    start = imag+1
    occupations_imag = np.array([np.fromstring(lines[start + i], sep=' ') for i in range(ndim)])
    occupations = occupations_real + 1j * occupations_imag
    return occupations


def plotrspt(text, l=3):
        plot_eigenvectors(*make_diagonal(parse_RSPT_matrix(text, l)), 'quanty', npts=50, fname=None); plt.show()


''' Reading and writing occupation matrices from quantum espresso '''
''' from the log file '''

def parse_eigen_data_4spin(text):
    '''
    Reading the 4spin hubbard matricies from the modified output of quantum espresso 
    This requires a patch to QE to get it to work:
        https://gist.github.com/ETrewick/bd995760a8e44b2617e8639f092a3a43
    Expects 'text' to be just a block of the normal output from QE, i.e.:

        ================= HUBBARD OCCUPATIONS ================
        ------------------------ ATOM    1 ------------------------
        Tr[ns(  1)] (up, down, total) =   6.48034  0.01465  6.49499
        eigenvalues:
          0.000  0.001  0.001  0.001  0.003  0.003  0.003  0.816  0.888  0.891  0.962  0.967  0.973  0.987
        eigenvectors (columns):
          0.000 -0.000 -0.000  0.000  0.000  0.000 -0.000  0.000  0.003  0.001  0.000 -0.000  0.000  0.000
          0.011 -0.000  0.004  0.000 -0.000 -0.002  0.000 -0.032  0.002 -0.108  0.001  0.386 -0.105 -0.000
         -0.006  0.000 -0.012  0.000  0.000  0.019 -0.000 -0.240 -0.008  0.576  0.000  0.540 -0.380  0.000
          0.000  0.000  0.000 -0.000 -0.000 -0.000  0.000  0.000 -0.000 -0.000 -0.000  0.000  0.000  0.000
         -0.000 -0.022 -0.000  0.000 -0.001  0.000 -0.000  0.000 -0.000 -0.000 -0.389  0.000  0.000 -0.921
          0.008  0.000 -0.003 -0.000 -0.000  0.000 -0.000  0.089 -0.002  0.073  0.000  0.141 -0.156 -0.000
          0.005  0.000 -0.008  0.000  0.000  0.001  0.000 -0.658 -0.006  0.388  0.000 -0.198  0.565 -0.000
         -0.000  0.000 -0.000 -0.000  0.000  0.122 -0.000 -0.001  0.000  0.000  0.000  0.015  0.000 -0.000
          0.000 -0.000 -0.000  0.000  0.000  0.000 -0.000 -0.000 -0.000 -0.000 -0.000  0.000  0.000 -0.000
         -0.000  0.559  0.001  0.565 -0.432  0.001 -0.425  0.000 -0.020 -0.000  0.007 -0.000 -0.000 -0.016
         -0.012 -0.001  0.333  0.000  0.000  0.000  0.000 -0.000 -0.000  0.004 -0.000 -0.000 -0.000 -0.000
         -0.504  0.000 -0.013 -0.000  0.000  0.000 -0.000  0.000  0.000 -0.007  0.000 -0.000  0.020 -0.000
          0.000 -0.000 -0.000  0.000 -0.001 -0.000  0.001 -0.000  0.000  0.000  0.000 -0.000  0.000  0.000
          0.000 -0.432 -0.000 -0.425 -0.559  0.001 -0.566 -0.000 -0.001 -0.000  0.013 -0.000 -0.000  0.005
        occupations, | n_(i1, i2)^(sigma1, sigma2) | real part |:
          0.887  0.000  0.000  0.000 -0.000 -0.000  0.000 -0.000 -0.017 -0.000 -0.000  0.000  0.001  0.000
          0.000  0.932  0.000 -0.000 -0.000  0.043  0.000  0.018 -0.000  0.000 -0.011 -0.000  0.000  0.000
          0.000  0.000  0.932  0.000  0.000 -0.000 -0.043 -0.000 -0.000 -0.000 -0.000 -0.012  0.000 -0.000
          0.000 -0.000  0.000  0.965  0.000 -0.000  0.000 -0.000  0.012 -0.000 -0.000 -0.000 -0.009 -0.000
         -0.000 -0.000  0.000  0.000  0.983  0.000  0.000  0.000  0.000  0.012  0.000  0.000 -0.000 -0.010
         -0.000  0.043 -0.000 -0.000  0.000  0.890 -0.000  0.000  0.000 -0.000  0.008  0.000  0.000 -0.000
          0.000  0.000 -0.043  0.000  0.000 -0.000  0.890 -0.000 -0.000  0.000 -0.000  0.009  0.000  0.000
         -0.000  0.018 -0.000 -0.000  0.000  0.000 -0.000  0.003 -0.000  0.000 -0.000  0.000  0.000  0.000
         -0.017 -0.000 -0.000  0.012  0.000  0.000 -0.000 -0.000  0.002  0.000  0.000 -0.000 -0.001  0.000
         -0.000  0.000 -0.000 -0.000  0.012 -0.000  0.000  0.000  0.000  0.002 -0.000 -0.000 -0.000  0.001
         -0.000 -0.011 -0.000 -0.000  0.000  0.008 -0.000 -0.000  0.000 -0.000  0.001  0.000  0.000 -0.000
          0.000 -0.000 -0.012 -0.000  0.000  0.000  0.009  0.000 -0.000 -0.000  0.000  0.000 -0.000  0.000
          0.001  0.000  0.000 -0.009 -0.000  0.000  0.000  0.000 -0.001 -0.000  0.000 -0.000  0.002 -0.000
          0.000  0.000 -0.000 -0.000 -0.010 -0.000  0.000  0.000  0.000  0.001 -0.000  0.000 -0.000  0.002
        occupations, | n_(i1, i2)^(sigma1, sigma2) | imag part |:
         -0.000 -0.000  0.000  0.000  0.000  0.000 -0.000  0.000 -0.000 -0.017  0.000  0.000 -0.000 -0.001
          0.000 -0.000 -0.016  0.000 -0.000 -0.000  0.005  0.000 -0.000 -0.000  0.000 -0.012  0.000 -0.000
         -0.000  0.016 -0.000 -0.000  0.000  0.005 -0.000  0.018  0.000 -0.000  0.011  0.000 -0.000 -0.000
         -0.000 -0.000  0.000 -0.000 -0.009  0.000  0.000  0.000  0.000 -0.012  0.000  0.000 -0.000 -0.009
         -0.000  0.000 -0.000  0.009 -0.000 -0.000 -0.000  0.000  0.012  0.000  0.000 -0.000  0.010  0.000
         -0.000  0.000 -0.005 -0.000  0.000  0.000 -0.056 -0.000 -0.000  0.000  0.000 -0.009  0.000  0.000
          0.000 -0.005  0.000 -0.000  0.000  0.056 -0.000 -0.000 -0.000 -0.000  0.008 -0.000 -0.000 -0.000
         -0.000 -0.000 -0.018 -0.000 -0.000  0.000  0.000 -0.000 -0.000  0.000  0.000  0.000 -0.000  0.000
          0.000  0.000 -0.000 -0.000 -0.012  0.000  0.000  0.000 -0.000  0.000 -0.000 -0.000  0.000  0.000
          0.017  0.000  0.000  0.012 -0.000 -0.000  0.000 -0.000 -0.000 -0.000 -0.000 -0.000  0.000  0.000
         -0.000 -0.000 -0.011 -0.000 -0.000 -0.000 -0.008 -0.000  0.000  0.000  0.000  0.000 -0.000  0.000
         -0.000  0.012 -0.000 -0.000  0.000  0.009  0.000 -0.000  0.000  0.000 -0.000 -0.000  0.000  0.000
          0.000 -0.000  0.000  0.000 -0.010 -0.000  0.000  0.000 -0.000 -0.000  0.000 -0.000 -0.000  0.000
          0.001  0.000  0.000  0.009 -0.000 -0.000  0.000 -0.000 -0.000 -0.000 -0.000 -0.000 -0.000  0.000
        Atomic magnetic moment mx, my, mz =    -0.000003   -0.000001    6.465695
    '''
    lines = text.strip().splitlines()
    
    start = lines.index("     eigenvalues:") + 1
    eigenvalues = np.fromstring(lines[start], sep=' ')
    ndim =  len(out['eigenvalues'])

    start = lines.index("     eigenvectors (columns):") + 1
    eigenvectors = np.array([np.fromstring(lines[start + i], sep=' ') for i in range(ndim)])

    start = lines.index("     occupations, | n_(i1, i2)^(sigma1, sigma2) | real part |:") + 1
    occupations_real = np.array([np.fromstring(lines[start + i], sep=' ') for i in range(ndim)])

    start = lines.index("     occupations, | n_(i1, i2)^(sigma1, sigma2) | imag part |:") + 1
    occupations_imag = np.array([np.fromstring(lines[start + i], sep=' ') for i in range(ndim)])

    occupations = occupations_real + 1j * occupations_imag

    return eigenvalues, eigenvectors, occupations




def parse_eigen_data_2spin(text):
    '''
    Reading the 2spin hubbard matricies from the modified output of quantum espresso 
    Example use:
        plot_eigenvectors_QEorder(*parse_eigen_data_2spin(text)[:2], npts=50, fname=None); plt.show()
    Expects 'text' to be just a block of the normal output from QE, e.g.:

         =================== HUBBARD OCCUPATIONS ===================
         ------------------------ ATOM    1 ------------------------
         Tr[ns(  1)] (up, down, total) =   5.06704  0.05798  5.12502
         Atomic magnetic moment for atom   1 =   5.00906
         SPIN  1
         eigenvalues:
           0.033  0.036  0.999  1.000  1.000  1.000  1.000
         eigenvectors (columns):
          -0.957  0.038  0.200 -0.041  0.137  0.149 -0.029
           0.146  0.052  0.546 -0.011  0.382 -0.292 -0.668
          -0.032  0.389 -0.257  0.120  0.648 -0.451  0.379
           0.205  0.028  0.029  0.105  0.537  0.810  0.022
           0.044  0.042 -0.113 -0.982  0.135  0.029  0.000
          -0.134 -0.387 -0.720  0.085  0.191 -0.051 -0.518
          -0.022  0.832 -0.253  0.032 -0.269  0.170 -0.375
         occupation matrix ns (before diag.):
           0.114  0.133 -0.044  0.188  0.040 -0.110 -0.051
           0.133  0.976 -0.015 -0.030 -0.008  0.038 -0.038
          -0.044 -0.015  0.853 -0.004 -0.014  0.141 -0.312
           0.188 -0.030 -0.004  0.958 -0.010  0.037 -0.018
           0.040 -0.008 -0.014 -0.010  0.996  0.021 -0.033
          -0.110  0.038  0.141  0.037  0.021  0.838  0.307
          -0.051 -0.038 -0.312 -0.018 -0.033  0.307  0.332
         SPIN  2
         eigenvalues:
           0.000  0.005  0.005  0.005  0.013  0.015  0.015
         eigenvectors (columns):
          -0.006 -0.009 -0.004 -0.026 -0.333 -0.941 -0.054
          -0.006 -0.334 -0.685  0.262 -0.513  0.166  0.244
          -0.000 -0.693  0.393  0.167  0.225 -0.109  0.524
          -0.001 -0.330 -0.192 -0.924  0.010  0.026  0.003
           1.000 -0.000 -0.000 -0.001 -0.010 -0.003  0.005
           0.008 -0.248 -0.502  0.194  0.688 -0.225 -0.355
          -0.001  0.487 -0.295 -0.112  0.319 -0.156  0.733
         occupation matrix ns (before diag.):
           0.014 -0.000  0.000 -0.000  0.000  0.000  0.000
          -0.000  0.008  0.000 -0.000  0.000 -0.004  0.000
           0.000  0.000  0.008 -0.000  0.000 -0.000  0.005
          -0.000 -0.000 -0.000  0.005  0.000 -0.000  0.000
           0.000  0.000  0.000  0.000  0.000 -0.000  0.000
           0.000 -0.004 -0.000 -0.000 -0.000  0.011 -0.001
           0.000  0.000  0.005  0.000  0.000 -0.001  0.011

    '''
    lines = text.strip().splitlines()
    spin_indices = (lines.index('     SPIN  1'), lines.index('     SPIN  2'))
    data = [{},{}]
    for spinstart, out in zip(spin_indices, data):
        start = lines.index('     eigenvalues:', spinstart) + 1
        out['eigenvalues'] = np.fromstring(lines[start], sep=' ')
        ndim =  len(out['eigenvalues'])
        start = lines.index('     eigenvectors (columns):', spinstart) + 1
        out['eigenvectors'] = np.array([np.fromstring(lines[start + i], sep=' ') for i in range(ndim)])
        start = lines.index('     occupation matrix ns (before diag.):', spinstart) + 1
        out['occupations'] = np.array([np.fromstring(lines[start + i], sep=' ') for i in range(ndim)])
    # build the 14x14 version of the above
    occupations = np.zeros((ndim*2,ndim*2),dtype=float)
    occupations[   0:ndim,      0:ndim  ] = data[0]['occupations'][:,:]
    occupations[ndim:ndim*2, ndim:ndim*2] = data[1]['occupations'][:,:]
    # it is technically possible to build the eigenvalues and eigenvectors matricies the same way.
    # I am too lazy as it's easier (to write) to re-diagonalise the occupations
    eigenvalues, eigenvectors = make_diagonal(occupations)
    return eigenvalues, eigenvectors, occupations

# testcase for the above: np.testing.assert_allclose(parse_eigen_data_2spin(parse_eigen_data_2spin.__doc__)[0], np.array([0.   , 0.004, 0.005, 0.005, 0.013, 0.014, 0.015, 0.034, 0.037, 0.998, 0.999, 0.999, 1.   , 1.   ]),atol=5e-4)


def plot2s(text):
        plot_eigenvectors(*parse_eigen_data_2spin(text)[:2], 'QE', npts=50, fname=None); plt.show()

def plot4s(text):
        plot_eigenvectors(*parse_eigen_data_4spin(text)[:2], 'QE', npts=50, fname=None); plt.show()




def read_token(buf):
    ''' python is a garbage language, and does not support reading a file by token delimited by anything other than a newline character
    '''
    token = bytearray()
    # skip leading whitespace
    while True:
        b = buf.peek(1)[:1]
        if not b:
            return ""
        if b not in b" \n\t\r":
            break
        buf.read(1)
    # read token
    while True:
        b = buf.peek(1)[:1]
        if not b or b in b" \n\t\r":
            break
        token += buf.read(1)
    return token.decode()

'''
    Functions to directly read and write the density matrix of QE (occup.txt)
    These functions are designed to be paired with 
    https://gist.github.com/ETrewick/0b4b484a2e680e94b11d3fe4ce74d27d
    Which allows the starting density matrix to be set with one from quanty
'''
def readocc_4spin(fname, ndim):
    # ndim = l*2+1
    # for 4f orbitals ndim = 4*2+1 = 7
    dens = np.zeros((ndim*2,ndim*2),dtype=complex)
    # d = np.genfromtxt(fname, dtype=complex, delimiter='%', max_rows=(ndim*2)**2,
    #                   converters={0: lambda s: complex(*map(np.float128, s.strip('()').split(',')))},
    #                   ).reshape((ndim*4,ndim))
    ctx = nullcontext(fname) if hasattr(fname, "write") else open(str(fname), "rb")
    d = []
    with ctx as f:
        for _ in range((ndim*2)**2):
            t = read_token(f)
            d.append(complex(*map(float, t[1:-1].split(','))) if t else None)
    d = np.array(d,dtype=complex).reshape((ndim*4,ndim))
    dens[   0:ndim,      0:ndim  ] = d[     0:ndim,  :]
    dens[ndim:ndim*2,    0:ndim  ] = d[  ndim:ndim*2,:]
    dens[   0:ndim,   ndim:ndim*2] = d[ndim*2:ndim*3,:]
    dens[ndim:ndim*2, ndim:ndim*2] = d[ndim*3:ndim*4,:]
    return(dens)

def writeocc_4spin(fname, dens):
    ndim = dens.shape[0] // 2
    d = np.array([
        dens[:ndim, :ndim],          # top-left
        dens[ndim:, :ndim],          # bottom-left
        dens[:ndim, ndim:],          # top-right
        dens[ndim:, ndim:]           # bottom-right
    ])
    # data = d.reshape((4*ndim, ndim))
    # data = np.vstack((d, np.zeros_like(d))).ravel()
    data = d.ravel() # it turns out that the zero matricies are writen out: all front, zeroes(all front), all back, zeroes(all back)
    np.savetxt(
        fname,
        data,
        fmt=" (%.15E,%.15E)",
        newline="\n"
    )
    
def readocc_2spin(fname, ndim):
    # ndim = l*2+1
    # for 4f orbitals ndim = 4*2+1 = 7
    dens = np.zeros((ndim*2,ndim*2),dtype=complex)
    # d = np.loadtxt(fname,max_rows=int(np.ceil(ndim*ndim*2*2/3))).ravel()[:ndim*ndim*2].reshape((ndim*2,ndim))
    ctx = nullcontext(fname) if hasattr(fname, "read") else open(str(fname), "rb")
    d = []
    with ctx as f:
        while len(d)< ndim*ndim*2:
            # d.extend(np.fromstring(f.readline(), sep=' '))
            d.append(float(read_token(f)))
    d = np.asarray(d)[:ndim*ndim*2].reshape((ndim*2,ndim))

    dens[   0:ndim,      0:ndim  ] = d[     0:ndim,  :]
    dens[ndim:ndim*2, ndim:ndim*2] = d[  ndim:ndim*2,:]
    return(dens)

def writeocc_2spin(fname, dens):
    ndim = dens.shape[0] // 2
    d = np.array([
        dens[:ndim, :ndim],          # top-left
        # dens[ndim:, :ndim],          # bottom-left
        # dens[:ndim, ndim:],          # top-right
        dens[ndim:, ndim:]           # bottom-right
    ])
    # data = d.reshape((2*ndim, ndim))
    # cast to real!!!!!!!!!!!
    # data = np.vstack((d, np.zeros_like(d))).ravel().real
    data = d.ravel().real # it turns out that the zero matricies are writen out: all front, zeroes(all front), all back, zeroes(all back)
    ctx = nullcontext(fname) if hasattr(fname, "write") else open(str(fname), "w")
    with ctx as f:
        for i in range(0, len(data), 3):
            row = data[i:i+3]
            f.write(" ".join(f"{x: .15E}" for x in row) + "\n")
    # np.savetxt(
    #     fname,
    #     data,
    #     fmt=" %.15E",
    #     newline="\n"
    # )

def readocc_4f_5d(fname):
    with open(fname, 'rb') as f:
        d_4f = readocc_2spin(f, 7)
        readocc_2spin(f, 7) # all zeroes
        d_5d = readocc_2spin(f, 5)
    return d_4f, d_5d

def writeocc_4f_5d(fname, d_4f, d_5d):
    with open(fname, 'w') as f:
        writeocc_2spin(f, d_4f)
        writeocc_2spin(f, d_4f*0.0) # write zeroes
        writeocc_2spin(f, d_5d)
        writeocc_2spin(f, d_5d*0.0)  # write zeroes

def load_quanty_density(dens_full_file, nstates=14):
    with open(dens_full_file,'r') as f:
        full_dens = np.loadtxt(f, dtype=complex, max_rows=nstates)
        f.readline()
        f.readline()
        occupations = np.loadtxt(f, dtype=complex, max_rows=1)
        f.readline()
        funcs_right = np.loadtxt(f, dtype=complex, max_rows=nstates)
    return full_dens, occupations, funcs_right


def make_CG_matrix(l=3,doPrint=True):
    doPrint and print('Generate Clebsch-Gordan matrix for f-orbitals')
    l= sp.S(1)/2
    ml_ms_basis = [(ml, ms) for ml in range(-l, l+1) for ms in [-s,  s]]
    j_mj_basis = [(j, mj) for j in [l-s, l+s] for mj in [sp.S(m) for m in np.arange(-j, j+1, 1)]]
    
    CG_matrix = sp.Matrix([[CG(l, ml, s, ms, j, mj).doit() 
                          for ml, ms in ml_ms_basis] 
                          for j, mj in j_mj_basis])
    
    doPrint and print(j_mj_basis)
    sp.pprint(CG_matrix)
    CG_mlms2jmj = np.asarray(CG_matrix).astype(float)
    doPrint and print(CG_mlms2jmj)
    
    return CG_mlms2jmj
    # # This is apparently in quanty order
    # CG_mlms2jmj = np.array([
    #         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0.37796447, 0., 0., 0.9258201, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0.53452248, 0., 0., 0.84515425, 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0.65465367, 0., 0., 0.75592895, 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0.75592895, 0., 0., 0.65465367, 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0.84515425, 0., 0., 0.53452248, 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9258201, 0., 0., 0.37796447],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    #         [0.9258201, 0., 0., 0.37796447, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., -0.84515425, 0., 0., 0.53452248, 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., -0.75592895, 0., 0., 0.65465367, 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., -0.65465367, 0., 0., 0.75592895, 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., -0.53452248, 0., 0., 0.84515425, 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.37796447, 0., 0., 0.9258201]])
    # 
    # # rearrange (ml,ms) to be (ms,ml)
    # CG_mlms2jmj = np.hstack((CG_mlms2jmj[:,::2], CG_mlms2jmj[:,1::2]))




def make_R_complex2real(l=3):
    ''' Rotation from complex spherical harmonics to real ones '''
    m = np.arange(-l, l + 1)
    U = np.zeros((2*l + 1, 2*l + 1), dtype=complex)
    
    for i, mi in enumerate(m):
        if mi == 0:
            U[i, m.tolist().index(0)] = 1
        elif mi > 0:
            U[i, m.tolist().index(-mi)] = 1/np.sqrt(2)
            U[i, m.tolist().index(mi)]  = (-1)**mi / np.sqrt(2)
        else:
            mp = -mi
            U[i, m.tolist().index(-mp)] = 1j / np.sqrt(2)
            U[i, m.tolist().index(mp)]  = -1j * (-1)**mp / np.sqrt(2)
    return U

def make_R_quanty2qe(L_max=3):
    ''' Build the rotation matrix from Quanty basis to QE basis '''
    # these are not the angular momenta (missing factor of 0.5 from spin) as they are only used for permutation
    # qe_orbs = [+0,+1,-1,+2,-2,+3,-3,+0,+1,-1,+2,-2,+3,-3]  # should be correct from the docs
    # qe_spin = [ 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1]
    qe_orbs = ([0] + [x for l in range(1, L_max+1) for x in (l, -l)])*2
    qe_spin = [1]*(L_max*2+1) + [-1]*(L_max*2+1)
    qe_order = list(zip(qe_orbs,qe_spin))
    # quanty_orbs = [-3,-3,-2,-2,-1,-1,-0,+0,+1,+1,+2,+2,+3,+3]
    # # quanty_spin = [-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1] # this one is technically correct
    # # but in quanty we have majority spin down, for some reason, so add a factor of -1 to spin
    # quanty_spin = [ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1]
    quanty_orbs = [m for m in range(-L_max, L_max+1) for _ in (0,1)]
    quanty_spin = [1, -1] * (2*L_max + 1)
    quanty_order = list(zip(quanty_orbs,quanty_spin))
    # build permutation matrix for reordering the angular momentum basis
    result = []
    for qe in qe_order:
        c = np.zeros(L_max*4+2)
        i = quanty_order.index(qe)
        c[i] = 1
        result.append(c)
    # mulitply permutation matrix by the real->complex spherical harmonics transfrom
    rot = np.asarray(result)@np.kron(make_R_complex2real(l=L_max), np.identity(2))
    # print(np.allclose(result.conj().T @ result, np.eye(14)))
    return rot

def convertquanty2qe(quanty_dens):
    R_quanty2qe = make_R_quanty2qe() 
    return R_quanty2qe@quanty_dens@R_quanty2qe.conj().T

def convertqe2quanty(qe_dens):
    R_quanty2qe = make_R_quanty2qe() 
    return R_quanty2qe.conj().T@qe_dens@R_quanty2qe

def make_diagonal(O):
    EO, RO = np.linalg.eig(O)
    sort_idx = np.argsort(EO.real)
    EOS = EO[sort_idx]
    ROS = RO[:, sort_idx]
    return EOS, ROS


def generate_ns_eigenvalues_for_desired_basis_occupations(initial_hubbard_text_data, desired=[1,1,1,.5,.5,.5,.5,0,0,0,0,0,0,0]):
    ''' generate ns_eigenvalues for desired basis occupations '''
    desired = np.asarray(desired)
    Ei, Ri, Oi = parse_eigen_data(initial_text_data)
    # Oi2 = Ri@np.diag(Ei)@Ri.T # = Oi (back to where we started)
    request = np.diagonal(Ri.T.conj()@np.diag(desired)@Ri).real

    print('desired', desired)
    print('request', request)
    return request


def plot_eigenvectors(occupations, funcs_right, format='QE', npts=50, fname=None, title=None, ftype='png', r_axis='R', c_axis='S', dens_r_axis='R', dens_c_axis='R'):
    ''' plots the eigenvectors that diagonalise a density matrix, and the whole density matrix
        format must be one of 'QE' or 'quanty'
        fname is the prefix of the output file
        title is the plot title, defaults to 'format'
        npts species the number of points in each direction of the angular mesh
    '''
    title = title or format
    c_axis = c_axis or r_axis
    L_max = (len(occupations)-2)//4  # ((len(occupations)/2)-1)/2
    if any([a not in ('R','L','S') for a in (r_axis, c_axis, dens_r_axis, dens_c_axis)]) :
        raise ValueError("r_axis, c_axis, dens_r_axis, dens_c_axis must be one of 'R','L','S'")
    if format == 'QE':
        # orbs = [ 0, 1,-1, 2,-2, 3,-3, 0, 1,-1, 2,-2, 3,-3]
        # spin = np.array([ 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1])/2
        orbs = ([0] + [x for l in range(1, L_max+1) for x in (l, -l)])*2
        spin = [0.5]*(L_max*2+1) + [-0.5]*(L_max*2+1)
        funcs_right = funcs_right.T
        def harm(ml, theta, phi):
            if ml < 0:
                return np.sqrt(2)*(-1)**(ml) * sph_harm_y(L_max, np.abs(ml), theta, phi).imag
            elif ml == 0:
                return sph_harm_y(L_max, 0, theta, phi).real
            elif ml > 0:
                return np.sqrt(2)*(-1)**(ml) * sph_harm_y(L_max, np.abs(ml), theta, phi).real
    elif format == 'quanty':
        # orbs = [-3,-3,-2,-2,-1,-1,-0,+0,+1,+1,+2,+2,+3,+3]
        # spin = np.array([ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1])/2
        orbs = [m for m in range(-L_max, L_max+1) for _ in (0,1)]
        spin = [0.5, -0.5] * (2*L_max + 1)
        def harm(ml, theta, phi):
            return sph_harm_y(L_max, ml, theta, phi)
    else:
        raise ValueError("format must be one of 'QE' or 'quanty'")
    cols_rows = {0:(2,2), 1:(3,3), 2:(4,3), 3:(5,3)}[L_max]
    fig, axes = plt.subplots(figsize=(10,6), ncols=cols_rows[0], nrows=cols_rows[1], subplot_kw={'projection': '3d'})
    
    theta = np.linspace(0, np.pi, npts)     # polar angle
    phi = np.linspace(0, 2 * np.pi, npts)   # azimuthal angle
    theta, phi = np.meshgrid(theta, phi)
    axlist = [ax for axrow in axes for ax in axrow]
    
    r_total = np.zeros_like(theta, dtype=complex)
    s_total = np.zeros_like(theta, dtype=complex)
    l_total = np.zeros_like(theta, dtype=complex)
    
    for ax, occ, vec in zip(axlist, occupations, funcs_right):
    
        ys = np.zeros_like(theta, dtype=complex)
        ss = np.zeros_like(theta, dtype=complex)
        ls = np.zeros_like(theta, dtype=complex)
        for ml, ms, coeff in zip(orbs, spin, vec):
            yc = coeff * harm(ml, theta, phi)
            ys += yc
            ss += ms*yc
            ls += ml*yc
        
        r = (ys.conj()*ys).real
        s = (ys.conj()*ss)  # not real and positive, but it's integral is
        l = (ys.conj()*ls)  # not real and positive, but it's integral is
        r_total += r.real*occ.real
        s_total += s*occ.real
        l_total += l*occ.real
        plotr = {'R':r,'S':s,'L':l}[r_axis]
        plotc = {'R':r,'S':s,'L':l}[c_axis]
        x = plotr * np.sin(theta) * np.cos(phi)
        y = plotr * np.sin(theta) * np.sin(phi)
        z = plotr * np.cos(theta)
    
        norm = plt.Normalize(vmin=-0.5, vmax=0.5)
        surf = ax.plot_surface(x, y, z,
                               rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.bwr(norm(plotc.real)),
                               antialiased=False)
        lim = 0.5
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        
        ax.set_title(f'eig:{occ.real:.3f}')
        ax.set_aspect('equal')
    
    ax = axlist[-1]    
    
    fig2, ax2 = plt.subplots(figsize=(5,5), ncols=1, nrows=1, subplot_kw={'projection': '3d'})
    for ax in (axlist[-1], ax2):
        plotr_total = {'R':r_total,'S':s_total,'L':l_total}[dens_r_axis]
        plotc_total = {'R':r_total,'S':s_total,'L':l_total}[dens_c_axis]
        x = plotr_total.real * np.sin(theta) * np.cos(phi)
        y = plotr_total.real * np.sin(theta) * np.sin(phi)
        z = plotr_total.real * np.cos(theta)
        
        norm = plt.Normalize(vmin=np.min(plotc_total.real), vmax=np.max(plotc_total.real))
        surf = ax.plot_surface(x, y, z,
                               rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.viridis(norm(plotc_total.real)),
                               antialiased=False)
        lim = max((np.max(np.abs(a)) for a in (x,y,z)))
        ax.set_xlim([-lim,lim])
        ax.set_ylim([-lim,lim])
        ax.set_zlim([-lim,lim])
        
        # ax.set_title(f'eig:{np.sum(occupations).real:.3f}')
        ax.set_aspect('equal')
    
    ax2.set_axis_off()
    fig2.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), ax=ax2)
    title and fig.suptitle(title)
    title and fig2.suptitle(title)
    # fig2.savefig(dens_full_file.split['.'][0] + '-charge.pdf')
    fname and fig2.savefig(fname + '-justdens.'+ftype, dpi=300)
    # fname and fig2.savefig(fname + '-charge_edit.png', dpi=200)
    fname and fig.savefig(fname + '-natural.'+ftype, dpi=300)
    
    return fig, fig2

def plot_density(full_dens, format='QE', npts=50, fname=None, title=None, ftype='png'):
    ''' plots the the whole density matrix
        format must be one of 'QE' or 'quanty'
        fname is the prefix of the output file
        title is the plot title, defaults to 'format'
        npts species the number of points in each direction of the angular mesh
    '''
    title = title or format
    L_max = (len(full_dens)-2)//4  # ((len(occupations)/2)-1)/2
    if format == 'QE':
        # orbs = [ 0, 1,-1, 2,-2, 3,-3, 0, 1,-1, 2,-2, 3,-3]
        # spin = np.array([ 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1])/2
        orbs = ([0] + [x for l in range(1, L_max+1) for x in (l, -l)])*2
        spin = [0.5]*(L_max*2+1) + [-0.5]*(L_max*2+1)
        funcs_right = funcs_right.T
        def harm(ml, theta, phi):
            if ml < 0:
                return np.sqrt(2)*(-1)**(ml) * sph_harm_y(L_max, np.abs(ml), theta, phi).imag
            elif ml == 0:
                return sph_harm_y(L_max, 0, theta, phi).real
            elif ml > 0:
                return np.sqrt(2)*(-1)**(ml) * sph_harm_y(L_max, np.abs(ml), theta, phi).real
    elif format == 'quanty':
        # orbs = [-3,-3,-2,-2,-1,-1,-0,+0,+1,+1,+2,+2,+3,+3]
        # spin = np.array([ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1])/2
        orbs = [m for m in range(-L_max, L_max+1) for _ in (0,1)]
        spin = [0.5, -0.5] * (2*L_max + 1)
        def harm(ml, theta, phi):
            return sph_harm_y(L_max, ml, theta, phi)
    else:
        raise ValueError("format must be one of 'QE' or 'quanty'")

    theta = np.linspace(0, np.pi, npts)     # polar angle
    phi = np.linspace(0, 2 * np.pi, npts)   # azimuthal angle
    theta, phi = np.meshgrid(theta, phi)
    fig3, ax3es = plt.subplots(figsize=(8,8), ncols=2, nrows=2, subplot_kw={'projection': '3d'})
    yt = np.zeros_like(theta, dtype=complex)
    st = np.zeros_like(theta, dtype=complex)
    lt = np.zeros_like(theta, dtype=complex)
    for i in range(len(full_dens)):
        for j in range(len(full_dens)):
            pij = full_dens[i,j]
            y = pij * harm(orbs[j], theta, phi).conj() * harm(orbs[i], theta, phi)
            yt += y
            st += y * spin[j]  # should be spin[j].conj(), but they're always real
            lt += y * orbs[j]  # should be orbs[j].conj(), but they're always real
    ax3list = [ax for axrow in ax3es for ax in axrow]
    for ax, pt, name in zip(ax3list, (yt.real, st.real, lt.real, (lt+2*st).real), (r'\rho',r's_z',r'l_z',r'm_z') ):
        x = pt * np.sin(theta) * np.cos(phi)
        y = pt * np.sin(theta) * np.sin(phi)
        z = pt * np.cos(theta)
        # norm = plt.Normalize(vmin=np.min(pt), vmax=np.max(pt))
        mmm = np.max(np.abs(pt))
        norm = plt.Normalize(vmin=-mmm, vmax=mmm)
        surf = ax.plot_surface(x, y, z,
                               rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.bwr(norm(pt)),
                               antialiased=False)
        # ax3.set_xlim([-0.5,0.5])
        # ax3.set_ylim([-0.5,0.5])
        # ax3.set_zlim([-0.5,0.5])
        ax.set_aspect('equal')
        ax.set_title('$'+name+r'(\hat{\mathbf{r}})$')
        
        
    fig3.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.bwr), ax=ax)
    title and fig3.suptitle(title)
    fname and fig3.savefig(fname + '-density.'+ftype, dpi=300)
    return fig3 



if __name__=="__main__":
    main()
