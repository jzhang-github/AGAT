# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:32:50 2021

@author: ZHANG Jun
"""

from pymatgen.core.structure import Structure
# from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.composition import Composition
import numpy as np

class AddAtoms(object):
    def __init__(self, fname, sites='all', species='O',
                 num_atomic_layer_along_Z=3, dist_from_surf=2.2, cutoff=5.0,
                 crystal_type='fcc'):
        self.fname                       = fname

        self._support_sites              = ['ontop', 'bridge', 'hollow']
        if isinstance(sites, str):
            sites = [sites]
        assert set(sites) < set(self._support_sites + ['all']), 'Illegal site name.'

        self._structure                  = Structure.from_file(self.fname)
        self.length_x                    = self._structure.lattice.a
        self.length_y                    = self._structure.lattice.b
        self.length_z                    = self._structure.lattice.c
        self.lattice_cell                = self._structure.lattice.matrix

        if sites == 'all':
            self.sites                   = self._support_sites
        else:
            self.sites                   = list(np.reshape([sites], (1,-1))[0])

        # r_range                          = 1.15
        # x_rand                           = np.random.uniform(-r_range, r_range)
        # y_rand                           = np.sqrt(1.3 ** 2 - x_rand ** 2) * [1.0, -1.0][np.random.randint(0, 2)]
        self.species                     = species
        self._relative_coords_of_species = {'O'  : {'O' : {'symbol': 'O', 'coord': self.cartesian2fractional(self.lattice_cell,
                                                                                                             np.array([0.0, 0.0, -0.2]))[0]}},
                                            'H'  : {'H' : {'symbol': 'H', 'coord': np.array([0.0, 0.0, -0.3])}},
                                            'OH' : {'O' : {'symbol': 'O', 'coord': np.array([0.0, 0.0, 0.0])},
                                                    'H' : {'symbol': 'H', 'coord': self.cartesian2fractional(self.lattice_cell,
                                                                                                             np.array([0.7, 0.0, 0.6]))[0]}},
                                            'OOH': {'O1': {'symbol': 'O', 'coord': self.cartesian2fractional(self.lattice_cell,
                                                                                                             np.array([0.0, 0.0, 0.0]))[0]},
                                                    'O2': {'symbol': 'O',
                                                           'coord' : self.cartesian2fractional(self.lattice_cell,
                                                                                               np.array([1.290, 0.0, 0.733]))[0]},
                                                    'H' : {'symbol': 'H',
                                                           'coord' : self.cartesian2fractional(self.lattice_cell,
                                                                                               np.array([1.290, 0.985, 0.733]))[0]}}
                                            }

        self.num_atomic_layers           = num_atomic_layer_along_Z
        self.dist_from_surf              = dist_from_surf
        self.cutoff                      = cutoff # this will be deprecated

        self.adsorbate_shift = {'bridge': 0.0 / self.length_z,
                                'ontop': 0.35 / self.length_z,
                                'hollow': -0.2 / self.length_z} # shift along z. Unit: angstrom

        self._Frac_coords                = self._structure.frac_coords
        max_z                            = np.max(self._Frac_coords, axis=0)[2]
        min_z                            = np.min(self._Frac_coords, axis=0)[2]
        layer_spacing                    = (max_z - min_z) / (num_atomic_layer_along_Z - 1)
        z_bottom                         = max_z - layer_spacing / 2
        self._top_layer_index            = np.where(self._Frac_coords[:,2] > z_bottom)

        self._top_layer_frac_coords      = self._Frac_coords[self._top_layer_index]


        self.crystal_type                = crystal_type
        crystal_density                  = {'fcc'  : 0.7404804896930611,
                                            'bcc'  : 0.6801747615878315,
                                            'hcp'  : 0.7404804896930611,
                                            'cubic': 0.5235987755982988}
        self.atomic_density              = crystal_density[self.crystal_type]
        self.atomic_diameter             = self.get_atomic_diameter_for_slab(layer_spacing,
                                                                             self.num_atomic_layers,
                                                                             self._structure)

        self.one_NN_top                  = self.get_1NN_of_top_layer(self._structure)

        self.recommend_dist_from_surface = {
                                             'NiCoFeAlTi': { 'O': 1.4, 'OOH': 0.0}
                                           }
    def shift_fcoords(self, fcoord1, fcoord2):
        diff        = fcoord1 - fcoord2
        transition  = np.where(diff >= 0.5, 1.0, 0.0)
        fcoord2_new = fcoord2 + transition
        transition  = np.where(diff < -0.5, 1.0, 0.0)
        fcoord1_new = fcoord1 + transition
        return fcoord1_new, fcoord2_new

    def get_middle_fcoord(self, fcoords):
        fcoords_tmp = fcoords.copy()
        num_coord = np.shape(fcoords_tmp)[0]
        for i in range(num_coord):
            for j in range(i + 1, num_coord):
                fcoords_tmp[i], fcoords_tmp[j] =\
                self.shift_fcoords(fcoords_tmp[i], fcoords_tmp[j])
        return np.sum(fcoords_tmp, axis=0) / num_coord

    def get_1NN_cutoff(self):
        node_dict = {'fcc'  : (1.0 + 1.4148706415278178) / 2,
                     'bcc'  : (1.0 + 1.1547005383792515) / 2,
                     'hcp'  : (1.0 + 1.4148706415278178) / 2,
                     'cubic': (1.0 + 1.4142135623730951) / 2}
        return node_dict[self.crystal_type] * self.atomic_diameter

    def get_atomic_diameter_for_slab(self, layer_sapcing, atomic_layers, structure):
        cell_volume   = structure.lattice.volume
        num_sites     = structure.num_sites
        effect_volume = layer_sapcing * atomic_layers * cell_volume
        return (effect_volume * self.atomic_density / num_sites * 3 / 4 / np.pi) ** (1 / 3) * 2

    def get_1NN_of_top_layer(self, structure=None):
        self.dist_mat = self._structure.lattice.get_all_distances(self._top_layer_frac_coords,
                                                             self._top_layer_frac_coords)
        OneNN_all = np.array(np.where(self.dist_mat < self.get_1NN_cutoff())).T
        return OneNN_all[np.where(OneNN_all[:,0] < OneNN_all[:,1])]

    def get_ontop_sites(self):
        addition = self.dist_from_surf / self.length_z
        ontop_coords_z = self._top_layer_frac_coords[:,2] + addition
        fcoords = np.hstack((self._top_layer_frac_coords[:,[0,1]], np.reshape(ontop_coords_z, (len(ontop_coords_z), -1))))
        fcoords[:,2] += self.adsorbate_shift['ontop']
        return fcoords

    def get_bridge_sites(self):
        fcoords  = []
        addition = self.dist_from_surf / self.length_z # * 0.8
        for i in self.one_NN_top:
            mid     = self.get_middle_fcoord(self._top_layer_frac_coords[[i[0], i[1]],:])
            mid[2] += addition
            fcoords.append(mid)
        fcoords = np.array(fcoords)
        fcoords[:,2] += self.adsorbate_shift['bridge']
        return fcoords

    def get_hollow_sites(self):
        addition   = self.dist_from_surf / self.length_z # * 0.6
        one_NN_top = self.one_NN_top.tolist()
        num_1nn    = len(one_NN_top)
        fcoord     = []
        for i in range(num_1nn):
            for j in range(i + 1, num_1nn):
                for k in range(j + 1, num_1nn):
                    nn_set = list(set(one_NN_top[i] + one_NN_top[j] + one_NN_top[k]))
                    if len(nn_set) == 3:
                        mid = self.get_middle_fcoord(self._top_layer_frac_coords[[nn_set[0], nn_set[1], nn_set[2]],:])
                        mid[2] += addition
                        fcoord.append(mid)
        fcoords = np.array(fcoord)
        fcoords[:,2] += self.adsorbate_shift['hollow']
        return fcoords

    def fractional2cartesian(self, vector_tmp, D_coord_tmp):
        C_coord_tmp = np.dot(D_coord_tmp, vector_tmp)
        return C_coord_tmp

    def cartesian2fractional(self, vector_tmp, C_coord_tmp):
        vector_tmp = np.mat(vector_tmp)
        D_coord_tmp = np.dot(C_coord_tmp, vector_tmp.I)
        D_coord_tmp = np.array(D_coord_tmp, dtype=float)
        return D_coord_tmp

    def write_file_with_adsorption_sites(self, file_format='Direct', calculation_index=0, partial_fix=False):
        site_func = dict(zip(self._support_sites, [self.get_ontop_sites, self.get_bridge_sites, self.get_hollow_sites]))

        frac_coords=[]
        for site in self.sites:
            frac_coords.append(site_func[site]())
        frac_coords = tuple(frac_coords)

        frac_coords = np.vstack(frac_coords)

        adsorbate_comp = Composition(self.species).as_dict()
        adsorbate_elem = list(adsorbate_comp.keys())
        adsorbate_num  = list(adsorbate_comp.values())

        comp           = self._structure.composition.as_dict()
        elements       = list(comp.keys())
        num_atom       = list(comp.values())

        elements_all   = elements + adsorbate_elem
        num_atom_all   = num_atom + adsorbate_num
        with open(self.fname, 'r') as f:
            lines = f.readlines()

        num_sites = np.shape(frac_coords)[0]
        for site_i in range(num_sites):
            # with open('POSCAR_' + str(site_i), 'w') as poscar:
            with open(f'POSCAR_{calculation_index}_{site_i}', 'w') as poscar:
                for i in elements_all:
                    poscar.write(str(i) + ' ')
                poscar.write('\n')
                for i in range(1,5):
                    poscar.write(lines[i])
                for i in elements_all:
                    poscar.write(str(i) + ' ')
                poscar.write('\n')
                for i in num_atom_all:
                    poscar.write(str(int(i)) + ' ')
                poscar.write('\n')
                poscar.write(f'Selective dynamics\n{file_format}\n')
                for i in range(int(np.sum(num_atom))):
                    poscar.write(lines[i+9])

            # new_structure = self._structure.copy()
                species = self._relative_coords_of_species[self.species]
                for ispecie_i, specie in enumerate(species):
                    adsorbate_coords = frac_coords[site_i] + species[specie]['coord']
                    if file_format == 'Cartesian':
                        adsorbate_coords = self.fractional2cartesian(self.lattice_cell, adsorbate_coords)

                    for i in adsorbate_coords:
                        poscar.write('  '+str(i))
                    if ispecie_i == 0 and partial_fix:
                        poscar.write('   F   F   T\n')
                    elif ispecie_i == 1 and partial_fix and specie == 'O2':
                        poscar.write('   F   F   T\n')
                    else:
                        poscar.write('   T   T   T\n')

        print('Number of POSCAR file generated:', num_sites)
        return num_sites

                # new_structure.append(Composition(species[specie]['symbol']), frac_coords[i] + species[specie]['coord'], properties={'selective_dynamics': [True, True, True]})
            # poscar = Poscar(new_structure)
            # fname = 'POSCAR_' + str(i)
            # poscar.write_file(fname)

if __name__ == '__main__':
    adder = AddAtoms('POSCAR.txt', species='OH', sites='bridge', dist_from_surf=2.0, num_atomic_layer_along_Z=5)
    num_sites = adder.write_file_with_adsorption_sites(file_format='Cartesian', partial_fix=True)
