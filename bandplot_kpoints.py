#!/usr/bin/python3
import numpy as np
import sys


def main():
    if len(sys.argv) == 1:
        # MAKE THIS WORK WITH CELLS OTHER THAN FCC!
        print('Ggenerate k points for band plots using QUANTUM EXPRESSO')
        print('Usage:')
        print('\tbandplot_kpoints.py NUM_POINTS CELL [UNITS [ROUTE]]')
        print('\tWhere NUM_POINTS is an integer')
        print('\tCELL is the name of a unit cell')
        print('\tROUTE is a string of labels of critical points')
        print("\t\t('|' can be used to create a dicontinuity between points)")
        print('\tUNITS is either "crystal" or "kvec"')
        print("\te.g.: ./bandplot_kpoints.py 100 FCC 'ΓXWKΓLUWLK|UX'\n")
        return
    if sys.argv[1].replace('.', '', 1).isdigit():
        NPts = float(sys.argv[1])
    else:
        print('First argument must be a number')
        return
    # if len(sys.argv) == 2:  # DIRTY HACKS! THIS "DEFAULT VALUE" SHOULD NOT BE
    #     sys.argv.append('FCC')
    #     sys.argv.append('ΓXWLΓK|UX')
    if (cell := sys.argv[2].upper()) not in (pC := set(NAME_TO_CRYSTAL_KVEC)):
        print(f'{cell} is not a known cell\n\tTry one of {pC}')
    if len(sys.argv) > 3:
        # clean the list of given critical points (excessivly complicated)
        route = sys.argv[3].upper().replace('G', 'Γ')
        Pkpts = set(NAME_TO_CRYSTAL_KVEC[cell])
        if len(route) <= 2:
            print('Please supply two or more labels from '
                  f'{Pkpts}')
            return
        wrong = set(route) - (Pkpts | set('|'))
        if wrong:
            if len(wrong) == 1:
                print(f'{str(wrong)[1:-1]} is not a valid point label.')
            else:
                print(f'{str(wrong)[1:-1]} are not valid point labels.')
            print(f'Try some of {Pkpts} instead.')
            return
    else:
        # take the default route through the cell
        route = getRecommendedRoute(cell)
    if len(sys.argv) > 4:
        units = sys.argv[4]
    else:
        units = 'crystal'
 
    print(printKpoints(NPts, cell, units, route).replace('Γ', 'G'))


def kpoints(NPts, cell, units, route):
    thingFromName = crystalFromName if units == 'crystal' else kvecFromName
    # or use 'crystal_b', and change α = NAME_TO_CRYSTAL_KVEC[a][0]
    route += '|'  # if route[-1] != '|'   # ensure last character is a |
    path = []
    weight = 0
    for a, b in zip(route, route[1:]):
        if a == '|':
            continue
        # the distance should always be measured in kspace (I think)
        d = (np.linalg.norm(kvecFromName(a, cell)-kvecFromName(b, cell))
             if b != '|' else 0)
        path.append([a, thingFromName(a, cell), d])
        weight += d
    weight = NPts/weight
    for p in path:
        p[-1] = int(round(p[-1]*weight))
    return path


def printKpoints(NPts, cell, units, route):
    path = kpoints(NPts, cell, units, route)
    output = ''
    actualNPts = 0
    for n, α, w in path:
        output += f'{α[0]: .4f} {α[1]: .4f} {α[2]: .4f} {w: 4d}  ! {n}\n'
        actualNPts += w
    tpe = 'crystal_b' if units == 'crystal' else 'tpiba_b'
    output = (f'K_POINTS {tpe}  ! {cell} {route}\n'
              + f'{len(path)}  ! total points = {actualNPts}\n' + output)
    return output


def getAllPermutations(vec):
    def permute(i, v):
        if i == len(v)-1:
            vecs.append(v.copy())
            if v[i]:
                v2 = v.copy()
                v2[i] *= -1
                vecs.append(v2)
            return
        prev = None
        for j in range(i, len(v)):  # swap this index with all subsequent
            if (j > i and v[i] == v[j]) or prev == v[j]:
                continue  # unless we've already tried that combination
            prev = v[j]
            v[i], v[j] = v[j], v[i]  # swap order
            permute(i+1, v)
            if v[i]:  # if the index is not zero, do it with the negated index
                v[i] *= -1
                permute(i+1, v)
                v[i] *= -1
            v[i], v[j] = v[j], v[i]  # swap order back
    vecs = []
    permute(0, np.sort(np.abs(vec)))
    return vecs


def kvecFromName(s, BLat):
    if (BZ := NAME_TO_CRYSTAL_KVEC.get(BLat)) and (P := BZ.get(s.upper())):
        return P['kvec']


def crystalFromName(s, BLat):
    if (BZ := NAME_TO_CRYSTAL_KVEC.get(BLat)) and (P := BZ.get(s.upper())):
        return P['crystal']


def kvecFromCrystal(k, BLat):
    if ((BZ := CRYSTAL_TO_NAME_KVEC.get(BLat))
            and (P := BZ.get(tuple(np.sort(np.abs(k)))))):  # (((sufficiency)))
        return P['kvec']


def crystalFromKvec(c, BLat):
    if ((BZ := KVEC_TO_NAME_CRYSTAL.get(BLat))
            and (P := BZ.get(tuple(np.sort(np.abs(c)))))):
        return P['crystal']


def nameFromCrystal(k, BLat):
    if ((BZ := CRYSTAL_TO_NAME_KVEC.get(BLat))
            and (P := BZ.get(tuple(np.sort(np.abs(k)))))):
        return P['name']


def nameFromKvec(c, BLat):
    if ((BZ := KVEC_TO_NAME_CRYSTAL.get(BLat))
            and (P := BZ.get(tuple(np.sort(np.abs(c)))))):
        return P['name']


def getCells():
    return set(NAME_TO_CRYSTAL_KVEC.keys())


def getPoints(cell):
    return set(NAME_TO_CRYSTAL_KVEC[cell].keys())


RECOMMENDED_ROUTE = {
    'CUB': 'ΓXMΓRX|MR',
    'FCC': 'ΓXWKΓLUWLK|UX',
    'BCC': 'ΓHNΓPH|PN',
    'HEX': 'ΓMKΓALHA|LM|KM',
    'TET': 'ΓXMΓZRAZ|XR|MA',
    'BCT': 'ΓXMΓZPNZM|MP',
    'ORC': 'ΓXSYΓZURTZ|YT|UZ|SR'
}


def getRecommendedRoute(cell):
    return RECOMMENDED_ROUTE.get(cell)


"""
    Here kvec is in units of (2pi/a, 2pi/b, 2pi/c)
    Taken from QE: Points inside the Brillouin zone and
    https://lampx.tugraz.at/~hadley/ss1/bzones/
"""
NAME_TO_CRYSTAL_KVEC = {  # in crystal coordinates
    'CUB': {
        'Γ': {'crystal': np.array((0.0,   0.0,   0.0)),
              'kvec':    np.array((0.0,   0.0,   0.0))},
        'R': {'crystal': np.array((0.5,   0.5,   0.5)),
              'kvec':    np.array((0.5,   0.5,   0.5))},
        'X': {'crystal': np.array((0.0,   0.5,   0.0)),
              'kvec':    np.array((0.0,   0.5,   0.0))},
        'M': {'crystal': np.array((0.5,   0.5,   0.0)),
              'kvec':    np.array((0.5,   0.5,   0.0))}
    },
    'FCC': {
        'Γ': {'crystal': np.array((0.0,   0.0,   0.0)),
              'kvec':    np.array((0.0,   0.0,   0.0))},
        'X': {'crystal': np.array((0.0,   0.5,   0.5)),
              'kvec':    np.array((0.0,   1.0,   0.0))},
        'L': {'crystal': np.array((0.5,   0.5,   0.5)),
              'kvec':    np.array((0.5,   0.5,   0.5))},
        'W': {'crystal': np.array((0.25,  0.75,  0.5)),
              'kvec':    np.array((0.5,   1.0,   0.0))},
        'U': {'crystal': np.array((0.25,  0.625, 0.625)),
              'kvec':    np.array((0.25,  1.0,   0.25))},
        'K': {'crystal': np.array((0.375, 0.75,  0.375)),
              'kvec':    np.array((0.75,  0.75,  0.0))}
    },
    'BCC': {
        'Γ': {'crystal': np.array((0.0,   0.0,   0.0)),
              'kvec':    np.array((0.0,   0.0,   0.0))},
        'H': {'crystal': np.array((-0.5,  0.5,   0.5)),
              'kvec':    np.array((0.0,   0.0,   1.0))},
        'P': {'crystal': np.array((0.25,  0.25,  0.25)),
              'kvec':    np.array((0.5,   0.5,   0.5))},
        'N': {'crystal': np.array((0.0,   0.5,   0.0)),
              'kvec':    np.array((0.0,   0.5,   0.5))}
    },
    'HEX': {
        'Γ': {'crystal': np.array((0.0,   0.0,   0.0)),
              'kvec':    np.array((0.0,   0.0,   0.0))},
        'A': {'crystal': np.array((0.0,   0.0,   0.5)),
              'kvec':    np.array((0.0,   0.0,   0.5))},
        'K': {'crystal': np.array((2/3,   1/3,   0.0)),
              'kvec':    np.array((2/3,   0.0,   0.0))},
        'H': {'crystal': np.array((2/3,   1/3,   1/2)),
              'kvec':    np.array((2/3,   0,     1/2))},
        'M': {'crystal': np.array((0.5,   0,     0.0)),
              'kvec':    np.array((0.5,  -12**-.5, 0))},
        'L': {'crystal': np.array((0.5,   0.0,  0.5)),
              'kvec':    np.array((0.5,  -12**-.5, 0.5))}
    },
    'TET': {
        'Γ': {'crystal': np.array((0.0,   0.0,   0.0)),
              'kvec':    np.array((0.0,   0.0,   0.0))},
        'X': {'crystal': np.array((0.5,   0.0,   0.0)),
              'kvec':    np.array((0.5,   0.0,   0.0))},
        'M': {'crystal': np.array((0.5,   0.5,   0.0)),
              'kvec':    np.array((0.5,   0.5,   0.0))},
        'Z': {'crystal': np.array((0.0,   0.0,   0.5)),
              'kvec':    np.array((0.0,   0.0,   0.5))},
        'R': {'crystal': np.array((0.5,   0.0,   0.5)),
              'kvec':    np.array((0.5,   0.0,   0.5))},
        'A': {'crystal': np.array((0.5,   0.5,   0.5)),
              'kvec':    np.array((0.5,   0.5,   0.5))}
    },
    'BCT': {
        'Γ': {'crystal': np.array((0.0,   0.0,   0.0)),
              'kvec':    np.array((0.0,   0.0,   0.0))},
        'X': {'crystal': np.array((0.5,   0.0,   0.0)),
              'kvec':    np.array((0.5,   0.5,   0.0))},
        'Z': {'crystal': np.array((0.5,   0.5,  -0.5)),
              'kvec':    np.array((1.0,   0.0,   0.0))},
        'N': {'crystal': np.array((0.0,   0.5,   0.0)),
              'kvec':    np.array((0.5,   0.0,   0.5))},
        'P': {'crystal': np.array((0.25,  0.25,  0.25)),
              'kvec':    np.array((0.5,   0.5,   0.5))}
    },
    'ORC': {
        'Γ': {'crystal': np.array((0.0,   0.0,   0.0)),
              'kvec':    np.array((0.0,   0.0,   0.0))},
        'X': {'crystal': np.array((0.5,   0.0,   0.0)),
              'kvec':    np.array((0.5,   0.0,   0.0))},
        'Y': {'crystal': np.array((0.0,   0.5,   0.0)),
              'kvec':    np.array((0.0,   0.5,   0.0))},
        'Z': {'crystal': np.array((0.0,   0.0,   0.5)),
              'kvec':    np.array((0.0,   0.0,   0.5))},
        'T': {'crystal': np.array((0.0,   0.5,   0.5)),
              'kvec':    np.array((0.0,   0.5,   0.5))},
        'U': {'crystal': np.array((0.5,   0.0,   0.5)),
              'kvec':    np.array((0.5,   0.0,   0.5))},
        'S': {'crystal': np.array((0.5,   0.5,   0.0)),
              'kvec':    np.array((0.5,   0.5,   0.0))},
        'R': {'crystal': np.array((0.5,   0.5,   0.5)),
              'kvec':    np.array((0.5,   0.5,   0.5))}
    }
}
KVEC_TO_NAME_CRYSTAL = {}
CRYSTAL_TO_NAME_KVEC = {}
# populate the dictionary
for cell, N2CK_perCell in NAME_TO_CRYSTAL_KVEC.items():
    K2NC_perCell = {}
    C2NK_perCell = {}
    for ptName, ptDict in N2CK_perCell.items():
        kvec = tuple(np.sort(ptDict['kvec']))
        crystal = tuple(np.sort(ptDict['crystal']))
        K2NC_perCell[kvec] = {'name': ptName, 'crystal': ptDict['crystal']}
        C2NK_perCell[crystal] = {'name': ptName, 'kvec': ptDict['kvec']}
    KVEC_TO_NAME_CRYSTAL[cell] = K2NC_perCell
    CRYSTAL_TO_NAME_KVEC[cell] = C2NK_perCell


if __name__ == '__main__':
    main()
