#! /usr/bin/python3
import numpy as np
import argparse
import sys

ele = {'number':[ 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
        76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
        87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
        98, 99, 100, 101, 102, 103, 104, 105, 106,
        107, 108, 109, 110, 111, 112],
       'symbol': ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
        'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
        'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
        'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
        'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
        'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
        'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
        'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
        'Ds', 'Rg', 'Uub'],
       'name': ['Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron',
        'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon', 'Sodium',
        'Magnesium', 'Aluminum', 'Silicon', 'Phosphorus', 'Sulfur',
        'Chlorine', 'Argon', 'Potassium', 'Calcium', 'Scandium',
        'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron',
        'Cobalt', 'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium',
        'Arsenic', 'Selenium', 'Bromine', 'Krypton', 'Rubidium',
        'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum',
        'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver',
        'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium', 'Iodine',
        'Xenon', 'Cesium', 'Barium', 'Lanthanum', 'Cerium',
        'Praseodymium', 'Neodymium', 'Promethium', 'Samarium',
        'Europium', 'Gadolinium', 'Terbium', 'Dysprosium', 'Holmium',
        'Erbium', 'Thulium', 'Ytterbium', 'Lutetium', 'Hafnium',
        'Tantalum', 'Tungsten', 'Rhenium', 'Osmium', 'Iridium',
        'Platinum', 'Gold', 'Mercury', 'Thallium', 'Lead', 'Bismuth',
        'Polonium', 'Astatine', 'Radon', 'Francium', 'Radium',
        'Actinium', 'Thorium', 'Protactinium', 'Uranium', 'Neptunium',
        'Plutonium', 'Americium', 'Curium', 'Berkelium', 'Californium',
        'Einsteinium', 'Fermium', 'Mendelevium', 'Nobelium',
        'Lawrencium', 'Rutherfordium', 'Dubnium', 'Seaborgium',
        'Bohrium', 'Hassium', 'Meitnerium', 'Darmstadtium',
        'Roentgenium', 'Ununbiium'],
       'mass': [1.00797, 4.0026, 6.941, 9.01218, 10.81,
        12.011, 14.0067, 15.9994, 18.998403, 20.179,
        22.98977, 24.305, 26.98154, 28.0855, 30.97376, 32.06,
        35.453, 39.948, 39.0983, 40.08, 44.9559, 47.9,
        50.9415, 51.996, 54.938, 55.847, 58.9332, 58.7,
        63.546, 65.38, 69.72, 72.59, 74.9216, 78.96,
        79.904, 83.8, 85.4678, 87.62, 88.9059, 91.22,
        92.9064, 95.94, 98, 101.07, 102.9055, 106.4,
        107.868, 112.41, 114.82, 118.69, 121.75, 127.6,
        126.9045, 131.3, 132.9054, 137.33, 138.9055, 140.12,
        140.9077, 144.24, 145, 150.4, 151.96, 157.25,
        158.9254, 162.5, 164.9304, 167.26, 168.9342, 173.04,
        174.967, 178.49, 180.9479, 183.85, 186.207, 190.2,
        192.22, 195.09, 196.9665, 200.59, 204.37, 207.2,
        208.9804, 209, 210, 222, 223, 226.0254, 227.0278,
        232.0381, 231.0359, 238.029, 237.0482, 242, 243,
        247, 247, 251, 252, 257, 258, 250, 260, 261,
        262, 263, 262, 255, 256, 269, 272, 277]}

ele_by_symbol = {S: {'number': A, 'name': N, 'mass':M}
                 for A, S, N, M in zip(*ele.values())}
ele_by_number = {N: {'symbol': S, 'name': N, 'mass':M}
                 for A, S, N, M in zip(*ele.values())}

def get_flag_from_line(l, index=-1):
    return l.strip().split()[index].lower().translate(str.maketrans('', '', '(){}[]'))

def load_pw_in(fname):
    with open(fname) as f:
        for l in f:
            if l.strip().startswith('nat'):
                nat = int(get_flag_from_line(l))
                continue
            if l.strip().startswith('celldm(0)'):  # celldm(0) is in bohr
                alat = float(get_flag_from_line(l))*0.529177249
                continue
            if l.strip().startswith('A '):  # A is in angstrom
                alat = float(get_flag_from_line(l))
                continue
            if l.startswith('CELL_PARAMETERS'):
                cell_type = get_flag_from_line(l)
                cell_p = np.array((next(f)+next(f)+next(f)).split(),
                                  dtype=float).reshape(3,3)
                continue
            if l.startswith('ATOMIC_POSITIONS'):
                atomic_type = get_flag_from_line(l)
                atomic_p = np.array(''.join([next(f) for _ in range(nat)]).split(),
                                  ).reshape(nat, 4)
                atomic_p_n = np.array([ele_by_symbol[m]['number'] for m in atomic_p[:,0]])
                atomic_p_v = atomic_p[:,1:].astype(float)
                continue
    if cell_type == 'bohr':
        cell_p *= 0.529177249 # bohr to angsrom
    elif cell_type == 'alat':
        cell_p *= alat  # alat to angsrom
    # cell_p is now in angstrom
    if atomic_type == 'alat':
        atomic_p_v *= alat
    elif atomic_type == 'bohr':
        atomic_p_v *= 0.529177249 # bohr to angsrom
    elif atomic_type == 'crystal':
        atomic_p_v[:,:] = atomic_p_v@cell_p
    
    return cell_p, atomic_p_n, atomic_p_v

def load_pw_out(fname):
    with open(fname) as f:
        for l in f:
            if l.strip().startswith('number of atoms/cell'):
                nat = int(get_flag_from_line(l))
                continue
            if l.strip().startswith('lattice parameter'):  # in bohr
                alat = float(get_flag_from_line(l, index=-2))*0.529177249
                continue
            if l.strip().startswith('crystal axes:'):
                cell_p = np.array([next(f).strip().split()[3:6] for _ in range(3)],
                                  dtype=float)
                continue
            if l.strip().startswith('Cartesian axes'):
                next(f)
                next(f)
                atomic_p_n = []
                atomic_p_v = []
                for _ in range(nat):
                    ln = next(f).strip().split()
                    atomic_p_n.append(ln[1])
                    atomic_p_v.append(ln[-4:-1])
                atomic_p_n = np.array([ele_by_symbol[m]['number'] for m in atomic_p_n])
                atomic_p_v = np.array(atomic_p_v, dtype=float)
                continue

    return cell_p*alat, atomic_p_n, atomic_p_v*alat


def write_xsf(fname, cell_p, atom_n, atom_v):
    with open(fname, 'w') as f:
        f.write('# written by Ted\'s terrible script\n')
        f.write('CRYSTAL\n')
        f.write('PRIMVEC\n')
        for l1, l2, l3 in cell_p:
            f.write(f'{l1: 12.8f}    {l2: 12.8f}    {l3: 12.8f}\n')
        # technically should also give CONVVEC (the conventional lattice vectors)
        f.write('PRIMCOORD\n')
        f.write(f'{len(atom_n)} 1\n')
        for e, v1, v2, v3 in zip(atom_n, *atom_v.T):
            f.write(f'{e:3d}    {v1: 12.8f}    {v2: 12.8f}    {v3: 12.8f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert pw in/out file to xsf')
    parser.add_argument('outfile', help='Output file name')
    parser.add_argument('--inp', action='store', default=None,
                        help='PW input file')
    parser.add_argument('--outp', action='store', default=None,
                        help='PW output file')
    args = parser.parse_args()
    if args.inp and args.outp or not (args.inp or args.outp):
        print('Either specify a pw input file OR a pw output file (not both)\n')
        parser.print_help(sys.stderr)
        sys.exit(0)
    if args.inp:
        d = load_pw_in(args.inp)
    if args.outp:
        d = load_pw_out(args.outp)
    write_xsf(args.outfile, *d)
