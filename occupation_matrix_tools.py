import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from contextlib import nullcontext
import sympy as sp
from sympy.physics.quantum.cg import CG

np.set_printoptions(precision=3, suppress=True, linewidth=10000000)



def main():
    '''
        I'm not even removely consistent with names. The density matrix is sometimes called the occupations matrix (O)
        The "natural orbtials" are the matrix which diagonalises the density matrix and is often called eigenvectors (evecs or evecs_right), or R (for rotation matrix)
        The the occupations of this matrix are sometimes called occupations, or simply eigenvalues (evals, or E) 

        I can only say that I'm sorry, and that in literature it's just as bad (although most papers are somewhat internally consistent)

        For testing, this line is helpful:
    '''



    # load a starting point from QE (we need the 5d electron density matrix)
    # qe_occup_fname = '/home/ludoric/Documents/PhD_stuff/QE/forDavidTam/SmN_paw/out/SmN_paw.save/occup.txt'
    path_prefix='/home/ludoric/Documents/PhD_stuff/quanty/QE_calc_from_quanty/'
    qe_occup_fname = path_prefix+'SmN_paw/it2_edited-occup.txt'
    qe_dens_4f, qe_dens_5d = readocc_4f_5d(qe_occup_fname)
    # for plotting I have to diagonalise the QE one
    qe_E, qe_R = make_diagonal(qe_dens_4f)
    qe_figs = plot_eigenvectors(qe_E, qe_R, 'QE', npts=50, fname=None)
    

    # load the output from quanty
    quanty_dens_fname = path_prefix+'Dy_2_Density_matrix_full.txt'
    quanty_4f_dens, quanty_4f_evals, quanty_4f_evecs = load_quanty_density(quanty_dens_fname)
    qu_figs = plot_eigenvectors(quanty_4f_evals, quanty_4f_evecs, 'quanty', npts=50, fname=None)

    
    # note that the QE and Quanty arrays use different basis functions, so we must perform a rotation between them
    R_quanty2qe = make_R_quanty2qe()


    quanty_4f_dens_in_qe = R_quanty2qe@quanty_4f_dens@R_quanty2qe.conj().T
    # additionally QE wants this to be all real (loose all angular momentum
    quanty_4f_dens_in_qe_real = 0.5 * (quanty_4f_dens_in_qe + quanty_4f_dens_in_qe.conj())

    # check that the rotation and plotting works correctly
    qu_E, qu_R = make_diagonal(quanty_4f_dens_in_qe_real)
    qu_figs2 = plot_eigenvectors(qu_E, qu_R, 'QE', npts=50, fname=None, title='quanty in QE format')

    # write the bloody thing back out for QE
    qe_occup_out_fname = './occup.txt-output' # '~/Documents/PhD_stuff/QE/forDavidTam/SmN_paw/out/SmN_paw.save/occup.txt'
    writeocc_4f_5d(qe_occup_out_fname, quanty_4f_dens_in_qe_real, qe_dens_5d)


    # try not to cry when it doesn't work
    


    plt.show()






''' Reading and writing occupation matrices from quantum espresso '''
''' from the log file '''

def parse_eigen_data_4spin(text):
    ''' Reading the 4spin hubbard matricies from the modified output of quantum espresso 
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

     Number of occupied Hubbard levels =    6.4950

    '''
    lines = text.strip().splitlines()
    
    start = lines.index("     eigenvalues:") + 1
    eigenvalues = np.fromstring(lines[start], sep=' ')

    start = lines.index("     eigenvectors (columns):") + 1
    eigenvectors = np.array([np.fromstring(lines[start + i], sep=' ') for i in range(14)])

    start = lines.index("     occupations, | n_(i1, i2)^(sigma1, sigma2) | real part |:") + 1
    occupations_real = np.array([np.fromstring(lines[start + i], sep=' ') for i in range(14)])

    start = lines.index("     occupations, | n_(i1, i2)^(sigma1, sigma2) | imag part |:") + 1
    occupations_imag = np.array([np.fromstring(lines[start + i], sep=' ') for i in range(14)])

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
        plot_eigenvectors_QEorder(*parse_eigen_data_2spin(text)[:2], npts=50, fname=None); plt.show()

''' directly from the ouput data'''
def readocc_4spin(fname, ndim):
    # ndim = l*2+1
    # for 4f orbitals ndim = 4*2+1 = 7
    dens = np.zeros((ndim*2,ndim*2),dtype=complex)
    d = np.genfromtxt(fname, dtype=complex, delimiter='%', max_rows=(ndim*2)**2,
                      converters={0: lambda s: complex(*map(np.float128, s.strip('()').split(',')))},
                      ).reshape((ndim*4,ndim))
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
    data = np.vstack((d, np.zeros_like(d))).ravel()
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
    ctx = nullcontext(fname) if hasattr(fname, "write") else open(str(fname), "r")
    d = []
    with ctx as f:
        while len(d)< ndim*ndim*2*2:
            d.extend(np.fromstring(f.readline(), sep=' '))
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
    data = np.vstack((d, np.zeros_like(d))).ravel().real # cast to real!!!!!!!!!!!
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
    with open(fname, 'r') as f:
        return readocc_2spin(f, 7), readocc_2spin(f, 5)

def writeocc_4f_5d(fname, d_4f, d_5d):
    with open(fname, 'w') as f:
        writeocc_2spin(f, d_4f)
        writeocc_2spin(f, d_5d)

def load_quanty_density(dens_full_file):
    with open(dens_full_file,'r') as f:
        full_dens = np.loadtxt(f, dtype=complex, max_rows=14)
        f.readline()
        f.readline()
        occupations = np.loadtxt(f, dtype=complex, max_rows=1)
        f.readline()
        funcs_right = np.loadtxt(f, dtype=complex, max_rows=14)
    return full_dens, occupations, funcs_right


def make_CG_matrix(doPrint=True):
    doPrint and print('Generate Clebsch-Gordan matrix for f-orbitals')
    l, s = 3, sp.S(1)/2
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

def make_R_quanty2qe():
    ''' Build the rotation matrix from Quanty basis to QE basis '''
    # these are not the angular momenta (missing factor of 0.5 from spin) as they are only used for permutation
    qe_orbs = [+0,+1,-1,+2,-2,+3,-3,+0,+1,-1,+2,-2,+3,-3]  # should be correct from the docs
    qe_spin = [ 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1]
    qe_order = list(zip(qe_orbs,qe_spin))
    quanty_orbs = [-3,-3,-2,-2,-1,-1,-0,+0,+1,+1,+2,+2,+3,+3]
    # quanty_spin = [-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1] # this one is technically correct
    # but in quanty we have majority spin down, for some reason, so add a factor of -1 to spin
    quanty_spin = [ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1]
    quanty_order = list(zip(quanty_orbs,quanty_spin))
    # build permutation matrix for reordering the angular momentum basis
    result = []
    for qe in qe_order:
        c = np.zeros(14)
        i = quanty_order.index(qe)
        c[i] = 1
        result.append(c)
    # mulitply permutation matrix by the real->complex spherical harmonics transfrom
    rot = np.asarray(result)@np.kron(make_R_complex2real(l=3), np.identity(2))
    # print(np.allclose(result.conj().T @ result, np.eye(14)))
    return rot


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


def plot_eigenvectors(occupations, funcs_right, format='QE', npts=50, fname=None, title=None, r_axis='R', c_axis='S', dens_r_axis='R', dens_c_axis='R'):
    ''' plots the eigenvectors that diagonalise a density matrix, and the whole density matrix
        format must be one of 'QE' or 'quanty'
        fname is the prefix of the output file
        title is the plot title, defaults to 'format'
        npts species the number of points in each direction of the angular mesh
    '''
    title = title or format
    c_axis = c_axis or r_axis
    if any([a not in ('R','L','S') for a in (r_axis, c_axis, dens_r_axis, dens_c_axis)]) :
        raise ValueError("r_axis, c_axis, dens_r_axis, dens_c_axis must be one of 'R','L','S'")
    if format == 'QE':
        orbs = [ 0, 1,-1, 2,-2, 3,-3, 0, 1,-1, 2,-2, 3,-3]
        spin = np.array([ 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1])/2
        funcs_right = funcs_right.T
        def harm(ml, theta, phi):
            if ml < 0:
                return np.sqrt(2)*(-1)**(ml) * sph_harm_y(3, np.abs(ml), theta, phi).imag
            elif ml == 0:
                return sph_harm_y(3, 0, theta, phi).real
            elif ml > 0:
                return np.sqrt(2)*(-1)**(ml) * sph_harm_y(3, np.abs(ml), theta, phi).real
    elif format == 'quanty':
        orbs = [-3,-3,-2,-2,-1,-1,-0,+0,+1,+1,+2,+2,+3,+3]
        spin = np.array([ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1])/2
        def harm(ml, theta, phi):
            return sph_harm_y(3, ml, theta, phi)
    else:
        raise ValueError("format must be one of 'QE' or 'quanty'")
    fig, axes = plt.subplots(figsize=(14,5), ncols=5, nrows=3, subplot_kw={'projection': '3d'})
    
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
    fname and fig2.savefig(fname + '-charge_edit.pdf')
    fname and fig2.savefig(fname + '-charge_edit.png', dpi=200)
    
    return fig, fig2



# def plot_eigenvectors_QEorder(occupations, funcs_right, npts=50, fname=None, title='QE'):
#     ''' plots the eigenvectors that diagonalise a density matrix, and the whole density matrix
#         Assumes the vectors are in "QE order"
#     '''
#     fig, axes = plt.subplots(figsize=(14,5), ncols=5, nrows=3, subplot_kw={'projection': '3d'})
#     
#     theta = np.linspace(0, np.pi, npts)     # polar angle
#     phi = np.linspace(0, 2 * np.pi, npts)   # azimuthaSm_06_+167,+005_Density_matrix_XYZ.txtl angle
#     theta, phi = np.meshgrid(theta, phi)
#     orbs = [ 0, 1,-1, 2,-2, 3,-3, 0, 1,-1, 2,-2, 3,-3]
#     spin = [ 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1]
#     
#     
#     
#     axlist = [ax for axrow in axes for ax in axrow]
#     
#     r_total = np.zeros_like(theta, dtype=complex)
#     s_total = np.zeros_like(theta, dtype=complex)
#     
#     for ax, occ, vec in zip(axlist, occupations, funcs_right.T):
#         # yup = np.zeros_like(theta, dtype=complex)
#         # ydw = np.zeros_like(theta, dtype=complex)
#         # for ml, ms, coeff in zip(orbs, spin, vec):
#         #     y = coeff * sph_harm_y(3, ml, theta, phi)
#         #     if ms == +1:
#         #         yup += y
#         #     elif ms == -1:
#         #         ydw += y
#         # 
#         # r = np.abs(yup)**2 + np.abs(ydw)**2
#         # s = (np.abs(yup)**2 - np.abs(ydw)**2)
#     
#         ys = np.zeros_like(theta, dtype=complex)
#         ss = np.zeros_like(theta, dtype=complex)
#         ls = np.zeros_like(theta, dtype=complex)
#         for ml, ms, coeff in zip(orbs, spin, vec):
#             if ml < 0:
#                 harm = np.sqrt(2)*(-1)**(ml) * sph_harm_y(3, np.abs(ml), theta, phi).imag
#             elif ml == 0:
#                 harm = sph_harm_y(3, 0, theta, phi).real
#             elif ml > 0:
#                 harm = np.sqrt(2)*(-1)**(ml) * sph_harm_y(3, np.abs(ml), theta, phi).real
#             yc = coeff * harm
#             ys += yc
#             ss += ms*yc
#             ls += ml*yc
#         
#         r = (ys.conj()*ys).real
#         s = (ys.conj()*ss)  # not real and positive, but it's integral is
#         r_total += r.real*occ.real
#         s_total += s.real*occ.real
#         x = r * np.sin(theta) * np.cos(phi)
#         y = r * np.sin(theta) * np.sin(phi)
#         z = r * np.cos(theta)
#     
#         norm = plt.Normalize(vmin=-1, vmax=1)
#         # norm = plt.Normalize(vmin=c.min(), vmax=c.max())
#         surf = ax.plot_surface(x, y, z,
#                                rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.bwr(norm(s.real)),
#                                antialiased=False)
#         ax.set_xlim([-0.5,0.5])
#         ax.set_ylim([-0.5,0.5])
#         ax.set_zlim([-0.5,0.5])
#         
#         ax.set_title(f'eig:{occ.real:.3f}')
#         ax.set_aspect('equal')
#     
#     ax = axlist[-1]    
#     
#     # r_total = np.square(r_total/2)
#     fig2, ax2 = plt.subplots(figsize=(5,5), ncols=1, nrows=1, subplot_kw={'projection': '3d'})
#     for ax in (axlist[-1], ax2):
#         # r_total_total-=0.25
#         x = r_total * np.sin(theta) * np.cos(phi)
#         y = r_total * np.sin(theta) * np.sin(phi)
#         z = r_total * np.cos(theta)
#         
#         # norm = plt.Normalize(vmin=-1, vmax=1)
#         norm_s = plt.Normalize(vmin=np.min(s_total.real), vmax=np.max(s_total.real))
#         mmm = np.max(np.abs(s_total.real))
#         norm = plt.Normalize(vmin=-mmm, vmax=mmm)
#         surf = ax.plot_surface(x, y, z,
#                                # rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.bwr(norm(s_total.real)),
#                                # rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.viridis(norm_s(s_total.real)),
#                                rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.viridis(norm_s(r_total.real)),
#                                antialiased=False)
#         # print(np.max(s_total.real), np.min(s_total.real))
#         # lim = 0.55
#         lim = max((np.max(np.abs(a)) for a in (x,y,z)))
#         ax.set_xlim([-lim,lim])
#         ax.set_ylim([-lim,lim])
#         ax.set_zlim([-lim,lim])
#         
#         # ax.set_title(f'eig:{np.sum(occupations).real:.3f}')
#         ax.set_aspect('equal')
#     
#     ax2.set_axis_off()
#     fig2.colorbar(plt.cm.ScalarMappable(norm=norm_s, cmap=plt.cm.viridis), ax=ax2)
#     title and fig.suptitle(title)
#     title and fig2.suptitle(title)
#     # fig2.savefig(dens_full_file.split['.'][0] + '-charge.pdf')
#     # fig2.savefig(dens_full_file.split('.')[-2].replace('/','') + '-charge_edit.png', dpi=200)
#     fname and fig2.savefig(fname+ '-charge_edit.pdf')
#     fname and fig2.savefig(fname+ '-charge_edit.png', dpi=200)
# 
# 
#     return fig, fig2
# 
# 
# def plot_eigenvectors_quantyorder(occupations, funcs_right, npts=50, fname=None, title='Quanty'):
#     ''' plots the eigenvectors that diagonalise a density matrix, and the whole density matrix
#         Assumes the vectors are in "Quanty order"
#     '''
#     fig, axes = plt.subplots(figsize=(14,5), ncols=5, nrows=3, subplot_kw={'projection': '3d'})
#     
#     theta = np.linspace(0, np.pi, npts)     # polar angle
#     phi = np.linspace(0, 2 * np.pi, npts)   # azimuthaSm_06_+167,+005_Density_matrix_XYZ.txtl angle
#     theta, phi = np.meshgrid(theta, phi)
#     orbs = [-3,-3,-2,-2,-1,-1,-0,+0,+1,+1,+2,+2,+3,+3]
#     spin = np.array([ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1])/2
# 
#     axlist = [ax for axrow in axes for ax in axrow]
#     
#     r_total = np.zeros_like(theta, dtype=complex)
#     s_total = np.zeros_like(theta, dtype=complex)
#     
#     for ax, occ, vec in zip(axlist, occupations, funcs_right):
#         # yup = np.zeros_like(theta, dtype=complex)
#         # ydw = np.zeros_like(theta, dtype=complex)
#         # for ml, ms, coeff in zip(orbs, spin, vec):
#         #     y = coeff * sph_harm_y(3, ml, theta, phi)
#         #     if ms == +1:
#         #         yup += y
#         #     elif ms == -1:
#         #         ydw += y
#         # 
#         # r = np.abs(yup)**2 + np.abs(ydw)**2
#         # s = (np.abs(yup)**2 - np.abs(ydw)**2)
#     
#         ys = np.zeros_like(theta, dtype=complex)
#         ss = np.zeros_like(theta, dtype=complex)
#         ls = np.zeros_like(theta, dtype=complex)
#         for ml, ms, coeff in zip(orbs, spin, vec):
#             yc = coeff * sph_harm_y(3, ml, theta, phi)
#             ys += yc
#             ss += ms*yc
#             ls += ml*yc
#         
#         r = (ys.conj()*ys).real
#         s = (ys.conj()*ss)  # not real and positive, but it's integral is
#         r_total += r.real*occ.real
#         s_total += s*occ.real
#         x = r * np.sin(theta) * np.cos(phi)
#         y = r * np.sin(theta) * np.sin(phi)
#         z = r * np.cos(theta)
#     
#         norm = plt.Normalize(vmin=-0.5, vmax=0.5)
#         # norm = plt.Normalize(vmin=c.min(), vmax=c.max())
#         surf = ax.plot_surface(x, y, z,
#                                rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.bwr(norm(s.real)),
#                                antialiased=False)
# 
#         lim = 0.5
#         ax.set_xlim([-lim, lim])
#         ax.set_ylim([-lim, lim])
#         ax.set_zlim([-lim, lim])
#         
#         ax.set_title(f'eig:{occ.real:.3f}')
#         ax.set_aspect('equal')
#     
#     
#     ax = axlist[-1]    
#     # print(s_total.imag.max())
#     
#     # r_total = np.square(r_total/2)
#     fig2, ax2 = plt.subplots(figsize=(5,5), ncols=1, nrows=1, subplot_kw={'projection': '3d'})
#     for ax in (axlist[-1], ax2):
#         # r_total_total-=0.25
#         x = r_total.real * np.sin(theta) * np.cos(phi)
#         y = r_total.real * np.sin(theta) * np.sin(phi)
#         z = r_total.real * np.cos(theta)
#         
#         # norm = plt.Normalize(vmin=-1, vmax=1)
#         norm_s = plt.Normalize(vmin=np.min(r_total.real), vmax=np.max(r_total.real))
#         # mmm = np.max(np.abs(s_total.real))
#         # norm = plt.Normalize(vmin=-mmm, vmax=mmm)
#         surf = ax.plot_surface(x, y, z,
#                                # rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.bwr(norm(s_total.real)),
#                                # rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.viridis(norm_s(s_total.real)),
#                                rstride=1, cstride=1, linewidth=0, facecolors=plt.cm.viridis(norm_s(r_total.real)),
#                                antialiased=False)
#         lim = max((np.max(np.abs(a)) for a in (x,y,z)))
#         # lim = 0.55
#         ax.set_xlim([-lim,lim])
#         ax.set_ylim([-lim,lim])
#         ax.set_zlim([-lim,lim])
#         
#         # ax.set_title(f'eig:{np.sum(occupations).real:.3f}')
#         ax.set_aspect('equal')
#     
#     ax2.set_axis_off()
#     fig2.colorbar(plt.cm.ScalarMappable(norm=norm_s, cmap=plt.cm.viridis), ax=ax2)
#     title and fig.suptitle(title)
#     title and fig2.suptitle(title)
#     # fig2.savefig(dens_full_file.split['.'][0] + '-charge.pdf')
#     fname and fig2.savefig(fname + '-charge_edit.pdf')
#     fname and fig2.savefig(fname + '-charge_edit.png', dpi=200)
#     
#     return fig, fig2
    
def plot_density_quantyorder(full_dens, npts=50, fname=None, title='Quanty'):
    ''' plots the the whole density matrix
        Assumes the vectors are in "Quanty order"
    '''
    theta = np.linspace(0, np.pi, npts)     # polar angle
    phi = np.linspace(0, 2 * np.pi, npts)   # azimuthaSm_06_+167,+005_Density_matrix_XYZ.txtl angle
    theta, phi = np.meshgrid(theta, phi)
    orbs = [-3,-3,-2,-2,-1,-1,-0,+0,+1,+1,+2,+2,+3,+3]
    spin = np.array([ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1])/2
    
    # calculate the total spin density another way:
    fig3, ax3es = plt.subplots(figsize=(8,8), ncols=2, nrows=2, subplot_kw={'projection': '3d'})
    yt = np.zeros_like(theta, dtype=complex)
    st = np.zeros_like(theta, dtype=complex)
    lt = np.zeros_like(theta, dtype=complex)
    for i in range(14):
        for j in range(14):
            pij = full_dens[i,j]
            y = pij * sph_harm_y(3, orbs[j], theta, phi).conj() * sph_harm_y(3, orbs[i], theta, phi)
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
    fname and fig3.savefig(fname + '-charge.pdf')
    title and fig3.suptitle(title)
    return fig3 



if __name__=="__main__":
    main()
