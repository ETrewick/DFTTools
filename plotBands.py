#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import bandplot_kpoints as kp
from pathlib import Path
import re
# import mpl_sqrtaxis
from .multicoloredline import colored_line

mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['lines.linewidth'] = 1

def main():
    # dat = readBandFile('bands_out_GdN-FCC.bands')
    # dup = readBandFile('bands2_GdN-FCC-S1.bands')
    # ddown = readBandFile('bands2_GdN-FCC-S2.bands')
    # dup = readBandFile('bands-up_GdN-FCC.bands')
    # ddown = readBandFile('bands-down_GdN-FCC.bands')
    # plotBands(dup, ddown)
    # bandD = loadLotsOfData('./', './fermi.tsv', orbital='Gd-4f')
    # plotKvsHU(bandD, 'FCC', kpoint='Gamma', individual=True, orbital='Gd-4f')
    # thus if the Gd4f band is 7.8eV below the fermi level, U should be 8.4eV
    # (from GdN THIN FILMS: BULK AND LOCAL ELECTRONIC...)
    # plotBandsDos('../tooBig/hubbardOut/H_Gd4f_8.00_GdN-FCC.dos',
    #              '../tooBig/hubbardOut/H_Gd4f_8.00_GdN-FCC-S1.bands',
    #              '../tooBig/hubbardOut/H_Gd4f_8.00_GdN-FCC-S2.bands')
    # bandD = loadLotsOfData('./',
    #                        './fermi.tsv',
    #                        orbital='Gd-5d')
    # plotKvsHU(bandD, kpoint='X', individual=False, orbital='Gd-5d')
    # plotBandgap(bandD, orbital='Gd-5d')

    # bands = readBandFile('Brel.bands') - 13.9718
    # proj = readBandFile('Brel.bands.3') 
    
    import sys 
    if len(sys.argv)<=2:
        print('syntax: plotBands [Fermi Energy] [title] [path], (optional [prefix])')
        print("eg. 'python -m DFTTools_TedTrewick.plotBands 14.0699 GdN_B GdN-")
    fermi = float(sys.argv[1])
    path = sys.argv[2].strip() # 'ΓXWKΓLUWLK|UX'
    title = sys.argv[3].strip()
    if len(sys.argv) > 4:
        prefix = sys.argv[4].strip()
    else:
        prefix = ''
    _, ks, rap = readBandFile(prefix+'S1.bands.rap')
    dup = readBandFile(prefix+'S1.bands')[0] - fermi  # 8.0736 # 4.5181
    ddw = readBandFile(prefix+'S2.bands')[0] - fermi  # 8.0736 # 4.5181
    f, ax = plotBands(path, rap, ks, dup, ddw)
    # ax.set_xlim(ax.get_xticks()[0], ax.get_xticks()[5])
    # f.set_figwidth(3)
    # f.set_figheight(4)
    # ax.legend().set_visible(False)
    f.savefig(f'{title}_bands.pdf')
    plt.show()
    # bandD = loadLotsOfData('./', './fermi.tsv', orbital='4f')
    # rap, _ = readBandFile('GdN-HB1-v70_Gd-4f_10.0-5d_6.0_-S2.bands.rap')
    # f, ax = plotKvsHU(bandD, rap, 'ΓXWKΓ', kpoint='X', individual=False, orbital='Gd-4f')
    # f.savefig('GdN_HU1.pdf', bbox_inches='tight')
    # plt.show() # 7.8eV below the fermi level is Gd-4f=5.95eV
    return
    _, ks, rap = readBandFile('GdN_B-S1.bands.rap')
    dup = readBandFile('GdN_B-S1.bands')[0] - 13.9718
    ddw = readBandFile('GdN_B-S2.bands')[0] - 13.9718
    f, ax = plotBands('ΓXWKΓLUWLK|UX', rap, ks, dup, ddw)
    # f, ax = plotSpinorBands('ΓXWKΓLUWLK|UX', rap, bands, proj, ylim=(-6,6))
    # ax.set_ylim((-15, 15))
    f.savefig('GdN_B_Bands.pdf')
    plt.show()

    # SmN - Paramagnetic phase 1.3ev from fermi measured from (5d-sup 5d-sdw)



def readBandFile(fname):
    with open(fname, 'r') as f:
        nbnd = nks = 0
        line = f.readline().split()
        for i in range(len(line)):
            if 'nbnd' in line[i]:
                nbnd = int(line[i+1].strip(','))
            elif 'nks' in line[i]:
                nks = int(line[i+1].strip(','))

        isCritical = []
        dat = []
        ks = []
        for _ in range(nks):
            coords = f.readline().split()
            # skip the damm coordinates for now
            ks.append(np.array([float(x) for x in coords[:-1]]))
            isCritical.append(coords[-1])
            # coords has an extra value if the file type is '.rap'
            values = []
            while len(values) < nbnd:
                # I don't know what these values mean
                values.extend(f.readline().split())
            dat.append(np.array(values, float))
    if fname[-4:] == '.rap':
        # print(isCritical)
        return np.array(dat).T, np.array(ks), np.array(isCritical) == 'T'
    else:
        return np.array(dat).T, np.array(ks)


def readFermiFile(fname, index_col=0, sep='\t'):
    """
        The fermi file is a (usually) .tsv file
        Each row associates some paramter of a file name to a fermi energy
        The first line of the file must give the names for the columns
        The loaded file is returned as a nested dictionary against index_col
    """
    with open(fname, 'r') as f:
        names = f.readline().strip('\r\n').split(sep)
        # datd = {float(l.split(sep)[0]):
        #         {zip(names[1:],map(float, l.split(sep)[1:]))}
        #         for l in f.readlines()}
        dat = np.array([np.array(ln.strip('\r\n').split(sep))
                        for ln in f.readlines()])
    dat[dat == ''] = 'NaN'
    dat = dat.astype(float)
    # datd = {l[0]: {zip(names[1:], l[1:])} for l in dat}
    dname = names.pop(index_col)
    l1keys = dat[:, index_col]
    l2values = np.delete(dat, index_col, axis=1)
    datd = {l1k: dict(zip(names, l2v)) for l1k, l2v in zip(l1keys, l2values)}
    return dname, datd


def loadLotsOfData(bandFName, zeroFN=None, zeroN='FermiE', orbital='Gd4f'):
    if zeroFN is not None:
        _, fD = readFermiFile(zeroFN)
    bandD = {}
    for file in Path(bandFName).glob('*.bands'):
        fstr = str(file.as_posix())
        if 'S2' in fstr:
            spin = '↓'
        elif 'S1' in fstr:
            spin = '↑'
        else:
            print('Spin not known from file name, defaulting to ↑')
            spin = '↑'
        fparts = re.split('/|_|-', fstr)
        HU = float(fparts[fparts.index(orbital)+1])
        # s = fparts[2][4:8]
        # if s[-1] == '-':
        #     s = s[:-1]
        # HU = float(s)

        if zeroFN is None:
            bandD.setdefault(HU, {})[spin] = readBandFile(str(file))[0]
        else:
            bandD.setdefault(HU, {})[spin] = readBandFile(str(file))[0] - fD[HU][zeroN]
    return bandD


def readDOS(dosFName, ax=None):
    # energy, dosup, dosdw, idos
    with open(dosFName) as f:
        FermiE = float(f.readline().split()[-2])
        datdos = np.loadtxt(f, unpack=True)
        datdos[0] -= FermiE
    return FermiE, datdos


def removeDuplicateLabels(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def routelablesXPointsFromString(route, rap, ks):
    ''' Route strings look like: 'ΓXWKΓLUWLK|UX'
        rap, ks can be obtained from:
            _, ks, rap = readBandFile(prefix+'S1.bands.rap')
        returns xaxis, kpoints_x, kpoints_n
    '''
    # calculate distances along horizontal axis of plot
    dist = np.linalg.norm(np.diff(ks, axis=0, prepend=[[np.nan]*3]), axis=1)
    dist[0] = 0  # the nan one
    dist[1:][(rap[:-1] & rap[1:])] = 0 # points that are discontinuities
    xaxis = np.cumsum(dist)

    # find critical points, skipping discontinuities indices 
    kpoint_indices =  np.concatenate((rap[:-1] & (rap[:-1] ^ rap[1:]), [True]))
    kpoints_x = xaxis[kpoint_indices]

    
    kpoints_n = '%'.join(list(route)).replace('%|%','|').replace('%_%','_').split('%')
    kpoints_n = [f'${a}$' for a in kpoints_n]
    kpoints_n = [a.replace('Γ', r'\Gamma') for a in kpoints_n]
    
    # make the arrays the same length
    kpoints_n = kpoints_n[:len(kpoints_x)]
    kpoints_n += ['?']*(len(kpoints_x) - len(kpoints_n))
    return xaxis, kpoints_x, kpoints_n


def plotDOS(datdos, ax=None, onY=False, nolegend=False, colour=None):
    if ax:
        f = None
    else:
        f = plt.figure(figsize=(9, 6))
        f.tight_layout()
        ax = f.add_subplot()
    if onY:
        Yup = Ydw = datdos[0]
        Xup = datdos[1]
        Xdw = datdos[2]
    else:
        Xup = Xdw = datdos[0]
        Yup = datdos[1]
        Ydw = datdos[2]
    # energy, dosup, dosdw, idos
    if datdos.shape[0] == 4:
        ax.plot(Xdw, Ydw, ':b', zorder=2, label='Minority Spin (↓)')
    elif datdos.shape[0] != 3:
        print(f"HELP! datdos is the wrong shape! It's {datdos.shape}")
    ax.plot(Xup, Yup, colour or '-r', zorder=1,
            label=None if nolegend else 'Majority Spin (↑)')
    # ax.plot(datdos[0], datdos[3], '-b', zorder=1, label='idos')
    if onY:
        ax.axhline(y=0, ls='--', c='k', lw=1, zorder=0)
        ax.set_xticks([])
        ax.set_xlabel('DOS')
        ax.set_ylabel('Energy [eV]')
        ax.set_xlim(0, None)
        # ax.set_xscale('log')
    else:
        ax.axvline(x=0, ls='--', c='k', lw=1, zorder=0)
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('DOS')
        ax.set_yticks([])
        ax.set_ylim(0, None)
        # ax.set_yscale('log')
    # mpl.scale.register_scale(mpl_sqrtaxis.SquareRootScale)
    # ax.set_yscale('squareroot')
    # ax.set_yticks(np.arange(0, 9, 2)**2)
    # ax.set_yticks(np.arange(0, 8.5, 0.5)**2, minor=True)

    nolegend or ax.legend()
    return f, ax


def plotBands(route, rap, ks, datup, datdw=None, ax=None, nolegend=False, ylim=(-10, 10), figsize=(9, 6), col=('r','b')):
    if ax:
        f = None
    else:
        # f = plt.figure(figsize=(9, 6))
        # f.tight_layout()
        # ax = f.add_subplot()
        f, (ax) = plt.subplots(ncols=1, figsize=figsize, constrained_layout=True)

    horiz, kpoints_x, kpoints_n = routelablesXPointsFromString(route, rap, ks)
    for i in range(datup.shape[0]):
        ax.plot(horiz, datup[i], c=col[0], ls='-', zorder=1, label='Majority Spin (↑)')
    if datdw is not None:
        for i in range(datdw.shape[0]):
            ax.plot(horiz, datdw[i], c=col[1], ls=':', zorder=2,
                    label='Minority Spin (↓)')
    # # kpoints_x = []
    # # kpoints_n = []
    # # for h, l in zip(horiz, datup[:, :3]):
    # #     if kpoint := kp.nameFromKvec(l, cell):
    # #         kpoints_x.append(h)
    # #         kpoints_n.append(kpoint)

    # drop = np.diff(np.where(rap)).ravel()
    # kpoints_x = horiz[rap]
    # print(kpoints_x)
    # # kpoints_n = [*route.replace('|', '')]
    # # for d in np.where(drop < 2)[0]:
    # #     kpoints_n[d] += '|' + kpoints_n.pop(d+1)

   
    # kpoints_n = '%'.join(list(route)).replace('%|%','|').replace('%_%','_').split('%')
    # kpoints_n = [f'${a}$' for a in kpoints_n]
    # kpoints_n = [a.replace('Γ', r'\Gamma') for a in kpoints_n]

    # 
    # kpoints_x = kpoints_x[np.append(drop != True, True)]
    # # while '|' in kpoints_n:
    # #     i = kpoints_n.index('|')
    # #     kpoints_n.pop(i)
    # #     kpoints_n[i-1] += kpoints_n.pop(i)  # maybe I don't need this

    # kpoints_n = kpoints_n[:len(kpoints_x)]
    # kpoints_n += ['?']*(len(kpoints_x) - len(kpoints_n))

    ax.set_xticks(kpoints_x)
    ax.set_xticklabels(kpoints_n)
    ax.grid(axis='x', color='k', linewidth=1, linestyle='-')
    ax.tick_params(axis='x', color='k', width=1)  # length=16,
    ax.margins(x=0)
    ax.set_ylim(-10, 10)
    ax.tick_params(axis='y', color='k', width=1)
    ax.axhline(0, c='k', linewidth=1, zorder=0)
    ax.set_ylabel('Energy [eV]')
    if datdw is not None and not nolegend:
        removeDuplicateLabels(ax)
    return f, ax


def plotSpinorBands(route, rap, ks, dat, proj=None, ax=None, ylim=(-10, 10)):
    if ax:
        f = None
    else:
        f, (ax) = plt.subplots(ncols=1, figsize=(9, 6))
        f.tight_layout()

    horiz, kpoints_x, kpoints_n = routelablesXPointsFromString(route, rap, ks)

    norm = mpl.colors.Normalize(-0.5, 0.5, clip=True)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('', ['blue','black','red'])
    for i, (ys, projs) in enumerate(zip(dat, np.zeros_like(dat) if proj is None else proj)):
        # # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
        # points = np.array([horiz, ys]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # lc = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm)
        # lc.set_array(projs[:-1])
        # lc.set_linewidth(2)
        # lc.set_joinstyle('round')
        # line = ax.add_collection(lc)
        # ax.scatter(horiz, ys, s=2, marker='.', c=projs, cmap=cmap, norm=norm, zorder=i, ) 
        colored_line(horiz, ys, c=projs, ax=ax, cmap=cmap, norm=norm, zorder=i)
    # f.colorbar(line, ax=ax)
    
    # kpoints_x = horiz[rap]
    # kpoints_n = [*route.replace('|', '')]
    # drop = np.diff(np.where(rap)).ravel()
    # for d in np.where(drop < 2)[0]:
    #     kpoints_n[d] += '|' + kpoints_n.pop(d+1)
    # kpoints_x = kpoints_x[np.append(drop != True, True)]

    # kpoints_n = kpoints_n[:len(kpoints_x)]
    # kpoints_n += ['?']*(len(kpoints_x) - len(kpoints_n))

    ax.set_xticks(kpoints_x)
    ax.set_xticklabels(kpoints_n)

    ax.grid(axis='x', color='k', linewidth=1, linestyle='-')
    ax.tick_params(axis='x', color='k', width=1)  # length=16,
    ax.margins(x=0)
    ax.set_ylim(ylim)
    ax.tick_params(axis='y', color='k', width=1)
    ax.axhline(0, c='k', linewidth=1, zorder=0)
    ax.set_ylabel('Energy [eV]')
    return f, ax


def plotKvsHU(bandD, rap, route, kpoint='Γ', orbital='Gd4f', individual=False,
              ax=None):

    lbls = np.empty(len(rap), dtype='U3')
    lbls[rap] = np.array([*route.replace('|', '')])
    # lbl_idx = np.nonzero(lbls)[0]
    # for d in np.nonzero(np.diff(lbl_idx) < 2)[0]:
    #     lbls[d] += '|' + lbls[d+1]
    #     lbls[d+1] = ''
    kindex = np.nonzero(lbls == kpoint)[0][0]

    Sup, Sdw, Hs = [], [], []
    for HU, S in sorted(bandD.items()):
        if individual:
            plotBands(route, S['↑'], S.get('↓'))
        Sup.append(S['↑'][:, kindex])
        Sdw.append(S['↓'][:, kindex])
        Hs.append(HU)
    Sup, Sdw, Hs = np.array(Sup), np.array(Sdw), np.array(Hs)
    # # sort all three arrays by the Hubbard U parameter
    # arginds = Hs.argsort()
    # Sup, Sdw, Hs = Sup[arginds], Sdw[arginds], Hs[arginds]
    # (already sorted by calling sorted(bandD.items()))
    if not ax:
        f = plt.figure(figsize=(4, 6), constrained_layout=True)
        ax = f.add_subplot()
    ax.plot(Hs, Sup, '-r', zorder=1, label='Majority Spin (↑)')
    ax.plot(Hs, Sdw, ':b', zorder=2, label='Minority Spin (↓)')
    removeDuplicateLabels(ax)
    ax.set_ylim(-10, 10)
    ax.set_ylabel(f'Energy at {kpoint} [eV]')
    ax.set_xlabel(f'Hubbard U for {orbital} [eV]')
    ax.margins(x=0)
    return f, ax


def plotBandgap(bandD, orbital='Gd4f', ax=None):
    Sup, Sdw, Hs = [], [], []
    for HU, S in sorted(bandD.items()):
        vup = S['↑'][:, 3:]
        vdw = S['↓'][:, 3:]
        sup = vup[vup > 0].min() - vup[vup < 0].max()
        sdw = vdw[vdw > 0].min() - vdw[vdw < 0].max()
        Sup.append(sup)
        Sdw.append(sdw)
        Hs.append(HU)
    Sup, Sdw, Hs = np.array(Sup), np.array(Sdw), np.array(Hs)
    # # sort all three arrays by the Hubbard U parameter
    # arginds = Hs.argsort()
    # Sup, Sdw, Hs = Sup[arginds], Sdw[arginds], Hs[arginds]
    # (already sorted by calling sorted(bandD.items()))

    if not ax:
        f = plt.figure(figsize=(9, 6), constrained_layout=True)
        ax = f.add_subplot()
    ax.plot(Hs, Sup, '-r', zorder=1, label='Majority Spin (↑)')
    ax.plot(Hs, Sdw, ':b', zorder=2, label='Minority Spin (↓)')
    # ax.set_ylim(-10, 10)
    ax.set_ylabel(f'Energy at [eV]')
    ax.set_xlabel(f'Hubbard U for {orbital} [eV]')
    ax.margins(x=0)
    return f, ax


def plotBandsDos(cell, dosFN, bandSupFN, bandSdwFN):
    FermiE, datdos = readDOS(dosFN)
    dup = readBandFile(bandSupFN) - FermiE
    ddown = readBandFile(bandSdwFN) - FermiE
    f, (ax1, ax2) = plt.subplots(
        ncols=2, sharey=True, figsize=(12, 6),
        gridspec_kw={'width_ratios': [3, 1]},
        constrained_layout=True)
    plotBands(cell, dup, ddown, ax=ax1, nolegend=True)
    plotDOS(datdos, ax=ax2, onY=True)
    ax2.set_ylabel(None)
    f.subplots_adjust(wspace=0)
    return f, (ax1, ax2)


if __name__ == '__main__':
    main()

