#! /usr/bin/python3
from lxml import etree, objectify
from scipy.ndimage import gaussian_filter
# my matplotlib version seems a little broken, so I have to import it multiple times
import matplotlib.patches as mpatches
import matplotlib.axis as maxis
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glob
import time
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.directory'] = '../'

plt.style.use('/storage/Ted-REN_Data/thesis_plot_style.mplstyle')
# from ted_helper_functions.py:
FIGSIZE = {'col_w': (204.9466/72.27)*1.5, # actually 0.49 of page width
           'page_w': (418.25368/72.27)*1.5
           } # for two column document
FIGSIZE = {'margin_w': (128.26378/72.27)*1.5,
           'page_w': (384/72.27)*1.5
           } # This is the one used in my first year report
FIGSIZE['column_w'] = FIGSIZE['page_w']*0.75
# FIGSIZE['col'] = (FIGSIZE['col_w'], FIGSIZE['col_w']*2/3)
FIGSIZE['page'] = (FIGSIZE['page_w'], FIGSIZE['column_w']*2/3)


# mpl.use('pgf') # "fix" for unicode charaacters


### you shouldn't have to edit this stuff:
def load_bands_xml(*, nspin, nbnd, nkpt, orb_fnames): 
    norb = len(orb_fnames)
    
    E = np.zeros((nspin, nbnd, nkpt))
    P = np.zeros((nspin, nbnd, nkpt))
    AP = np.zeros((norb, nspin, nbnd, nkpt))
    K = np.zeros((nkpt, 3))
    for i, orb in enumerate(orb_fnames):
        for fname in glob.glob(f'{orb}.dat.save/*.xml'):
            with open(fname,'r') as f:
                xml = f.read()
            data = objectify.fromstring(xml)
            ik = int(data.EIGENVALUES.attrib['ik'])
            ispin = int(data.EIGENVALUES.attrib['ispin'])
            if i == 0:
                E[ispin-1, :, ik-1]  = data.EIGENVALUES.text.split()
                P[ispin-1, :, ik-1]  = data.PROJECTIONS.text.split()
                K[ik-1, :] = data['K-POINT_COORDS'].text.split()
            AP[i, ispin-1, :, ik-1] = data.ATOMIC_PROJECTIONS.text.split()
    return E, P, AP, K

def save_bands_npy(*, E, P, AP, K, fname='rawbands.npz'):
    np.savez_compressed(fname, E=E, P=P, AP=AP, K=K)

def load_bands_npy(*, fname='rawbands.npz'):
    with np.load(fname) as d:
        return d['E'], d['P'], d['AP'], d['K']



def eventplot(E, P, AP, *, spin=1):
    ''' simple, and poor, plot function'''
    f, ax = plt.subplots()
    for i in range(AP.shape[0]):
        ax.eindeciesventplot(E[spin-1].T, orientation='vertical',
                             linewidth=(10*AP[i, spin-1].T*P[spin-1].T),
                             colors=plt.cm.tab10(i))
    return f


def rasterise(*, E, P, AP, Emax, Emin, DeltaE, Fermi):
    ''' Rasterise data
        R.shape = (norb, nspin, nk, nenergy)
    '''
    e_count = int((Emax-Emin)/DeltaE) # total number of pixels
    # AP.shape = (norb, nspin, nbnd, nk)
    R = np.zeros((AP.shape[0], AP.shape[1], AP.shape[3], e_count))
    # lastE = lastPAP = None
    # for ind in np.ndindex(NORBG, 2, nbnd, nkpt):
    #     e = E[ind]
    #     pap = P[ind]*AP[ind]
    #     ei, ef = np.divmod(e-FERMI-Emin, DeltaE)
    #     if ibnd == 0 or ik in critical_indecies or ei < 0.0 or ei+1.0 < e_cout:
    #         lastE = e
    #         lastPAP = pap
    #         continue
    #     i, ispin, ibnd, ik = ind
    #     for kp in range(kwidth):
    #         R[i, ispin, ik+kp, int(ei)] += ef*pap
    #         R[i, ispin, ik+kp, int(ei)+1] += (1.0-ef)*pap
    # start = time.perf_counter_ns()
    # for ind in np.ndindex(E.shape):
    #     ei, ef = np.divmod(E[ind]-Fermi-Emin, DeltaE)
    #     if ei < 0.0 or ei+1.0 >= e_count:
    #         continue
    #     p = P[ind]
    #     ispin, ibnd, ik = ind
    #     for i in range(AP.shape[0]):
    #         pap = p*AP[i, ispin, ibnd, ik]
    #         R[i, ispin, ik, int(ei)] += ef*pap
    #         R[i, ispin, ik, int(ei)+1] += (1.0-ef)*pap
    # print(E.shape)
    # E = E.transpose((0,2,1))
    # ei, ef = np.divmod(E.ravel()-Fermi-Emin, DeltaE)
    # ei = ei.astype(int)
    # wh = (ei < 0) & (ei+1 >= e_count)
    # indexes = np.repeat(np.arange(len(E[:,:].ravel()))*e_count, E.shape[-1])
    # for i in range(AP.shape[0]):
    #     pap = (P*AP[i]).transpose(0,2,1).ravel()
    #     R.ravel()[indexes+ei] += ef*pap
    #     R.ravel()[indexes+ei+1] += (1.0-ef)*pap
    

    for ind in np.ndindex(E.shape):
        ei, ef = np.divmod(E[ind]-Fermi-Emin, DeltaE)
        if ei < 0.0 or ei+1.0 >= e_count:
            continue
        p = P[ind]
        ispin, ibnd, ik = ind
        for i in range(AP.shape[0]):
            pap = p*AP[i, ispin, ibnd, ik]
            R[i, ispin, ik, int(ei)] += ef*pap
            R[i, ispin, ik, int(ei)+1] += (1.0-ef)*pap

    # end = time.perf_counter_ns()
    # print(end-start)
    return R


def blur(*, R, blur_sigma, DeltaE, critical_indices):
    ''' Blur data
    blur_sigma: in units of k, eV respectively
    '''
    for i, ispin in np.ndindex(R.shape[:2]):
        for k1, k2 in zip(critical_indices[:-1], critical_indices[1:]):
            r = R[i, ispin, k1:k2]
            if r.shape[1] == 1:
                continue
            gaussian_filter(
                    r, sigma=(blur_sigma[0], blur_sigma[1]/DeltaE),
                    output=r, mode='constant')
    # don't return anything; occours in place


def scale01(V, minv, maxv):
    return np.clip((V-minv)/(maxv-minv), 0.0, 1.0, out=V)
    

def colour_multiply(*, R, colours):
    ''' convert band densities to colors, and combine orbital projections
        (simply multiplying them looks pretty good)
    '''
    c0 = (1.0, 1.0, 1.0)
    RGB = np.zeros((*R.shape, 3))
    for ind in np.ndindex(R.shape[:2]):
        c1 = mpl.colors.to_rgb(colours[ind])
        v = R[ind]
        RGB[ind] = np.multiply.outer(1.0-v, c0) + np.multiply.outer(v, c1)
    RGB_sum = np.multiply.reduce(RGB, axis=(0, 1))
    del RGB
    return RGB_sum


def colour_overlay(*, R, colours, orbder):
    '''
        orbder = order to plot the orbitals in
    '''
    def dumbcmap(v, c):
        # c0 = (1.0, 1.0, 1.0, 0.0)
        c1 = mpl.colors.to_rgba(c)
        c0 = (*c1[:-1], 0.0)
        return (np.multiply.outer(1.0-v, c0) + np.multiply.outer(v, c1))
    
    def addcolours(colours1, colours2):
        c0 = colours1.T
        c1 = colours2.T
        a = (1.0-c0[3:4])*c1[3:4] + c0[3:4]
        rgb = ((1.0-c0[3:4])*c1[3:4]*c1[:3] + c0[3:4]*c0[:3])/a
        return np.append(rgb.T, a.T, axis=-1)
    
    RGB_sum = np.ones((R.shape[2], R.shape[3], 4))
    for iorb in orbder:
        for ispin in range(R.shape[1]):
            RGB = dumbcmap(R[iorb, ispin], colours[iorb, ispin])
            RGB_sum = addcolours(RGB, RGB_sum)
    return RGB_sum


def plot_RGB(*, RGB, Emin, Emax, critical_indices, critical_labels,
             orb_labels=None, colours=None, K=None, nspin=None, showGrid=True, showZeroE=True):
    ''' plot '''
    fig, ax = plt.subplots(layout='constrained', figsize=(FIGSIZE['page_w'], 2/3*FIGSIZE['page_w']))
    ax.imshow(RGB.transpose((1, 0, 2)), origin='lower', aspect='auto',
              extent=(0, critical_indices[-1], Emin, Emax))
    # ax.margins(x=0, y=0)
    ax.set_xticks(critical_indices)
    ax.set_xticklabels(critical_labels)
    if showGrid:
        ax.grid(axis='x', color='k', linewidth=1, linestyle='-')
    if showZeroE:
        ax.axhline(0, c='k', linewidth=1, zorder=0)
    ax.tick_params(axis='x', color='k', width=1)  # length=16,
    ax.tick_params(axis='y', color='k', width=1)
    ax.set_ylabel('Energy [eV]')
    
    
    if K is not None:
        # fiddley bit to get some labels for the different directions
        ax2 = maxis.XAxis(ax)
        ax2.set_ticks(
                list((critical_indices[1:] + critical_indices[:-1])/2),
                [f'({h}, {k}, {l})' for h, k, l in K[critical_indices[1:]-1]])
        ax2.set_tick_params(
            top=False, bottom=False, left=False, right=False,
            labeltop=True, labelbottom=False, labelleft=False, labelright=False)
        ax.add_artist(ax2)
    
    if orb_labels is not None and colours is not None:
        # add colour legend
        spins = '↑↓' if nspin == 2 else ''
        patches = [mpatches.Patch(color=c, label=f'{o} {s}')
                   for cp, o in zip(colours, orb_labels) for c, s in zip(cp, spins)]
        fig.legend(handles=patches, loc='outside upper center', ncols=len(colours)*nspin)
    
    return fig


# def addcolours(c0, c1):
#     # a = (1.0-c0[3])*c1[3] + c0[3]
#     # rgb = ((1.0-c0[3])*c1[3]*c1[:3] + c0[3]*c0[:3])/a
#     # return np.stack(rgb, a)
#     ca0 = c0.copy()
#     ca1 = c1.copy()
#     ca1[:3] *= ca1[3]
#     ca1 *= 1.0-ca0[3]
#     ca0[:3] *= ca0[3]
#     ca1 += ca0
#     ca1[:3] /= ca1[3]
#     del ca0
#     return ca1
