# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:32:50 2021

@author: ZHANG Jun
"""

import numpy as np
from ase import Atoms, Atom
from ase.io import read, write
from ase.formula import Formula
from ase.build import add_adsorbate
from io import StringIO

class AddAtoms(object):
    def __init__(self, fname, sites='all', species='O',
                 num_atomic_layer_along_Z=3, dist_from_surf=2.2,
                 crystal_type='fcc'):
        self.fname                       = fname

        self._support_sites              = ['ontop', 'bridge', 'hollow', 'disigma']

        if isinstance(sites, str):
            self.sites = [sites]
        else:
            self.sites = sites
        if sites[0] == 'all':
            self.sites               = self._support_sites
        self.sites                   = list(np.reshape(self.sites, (1,-1))[0])

        assert set(self.sites) < set(self._support_sites + ['all']), 'Illegal site name.'

        self._structure                  = read(self.fname)
        # self.length_x                    = self._structure.cell.array[0][0]
        # self.length_y                    = self._structure.cell.array[1][1]
        self.length_z                    = self._structure.cell.array[2][2]
        self.lattice_cell                = self._structure.cell.array

        self.species                     = species

        self.num_atomic_layers           = num_atomic_layer_along_Z
        self.dist_from_surf              = dist_from_surf

        # self.adsorbate_shift = {'bridge': 0.0,
        #                         'ontop': 0.35,
        #                         'hollow': -0.2} # shift along z. Unit: angstrom

        self._Frac_coords                = self._structure.get_scaled_positions()
        self._Cart_coords                = self._structure.get_positions()
        max_z                            = np.max(self._Cart_coords, axis=0)[2]
        min_z                            = np.min(self._Cart_coords, axis=0)[2]
        layer_spacing                    = (max_z - min_z) / (num_atomic_layer_along_Z - 1)
        z_bottom                         = max_z - layer_spacing / 2
        self._top_layer_index            = np.where(self._Cart_coords[:,2] > z_bottom)

        self._top_layer_frac_coords      = self._Frac_coords[self._top_layer_index]
        self._top_layer_cart_coords      = self.fractional2cartesian(self.lattice_cell,
                                                                     self._Frac_coords[self._top_layer_index])

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

        # self.recommend_dist_from_surface = {
        #                                      'NiCoFeAlTi': { 'O': 1.4, 'OOH': 0.0}
        #                                    }

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
        cell_volume   = structure.get_volume()
        num_sites     = len(structure)
        effect_volume = layer_sapcing / self.length_z * atomic_layers * cell_volume
        return (effect_volume * self.atomic_density / num_sites * 3 / 4 / np.pi) ** (1 / 3) * 2

    def get_1NN_of_top_layer(self, structure=None):
        atoms_new = self._structure.copy()
        del atoms_new[:]
        for ccoord in self._top_layer_cart_coords:
            atoms_new.append(Atom('H', ccoord))
        self.dist_mat = atoms_new.get_all_distances(mic=True)
        OneNN_all = np.array(np.where(self.dist_mat < self.get_1NN_cutoff())).T
        return OneNN_all[np.where(OneNN_all[:,0] < OneNN_all[:,1])]

    def get_ontop_sites(self, x_shift=0.0, y_shift=0.0):
        ontop_coords_z = self._top_layer_cart_coords[:,2]
        ccoords = np.hstack((self._top_layer_cart_coords[:,[0,1]], np.reshape(ontop_coords_z, (len(ontop_coords_z), -1))))
        ccoords[:,0] += x_shift
        ccoords[:,1] += y_shift
        return ccoords, np.array([None for x in ccoords])

    def get_bridge_sites(self, x_shift=0.0, y_shift=0.0):
        fcoords  = []
        for i in self.one_NN_top:
            mid     = self.get_middle_fcoord(self._top_layer_frac_coords[[i[0], i[1]],:])
            fcoords.append(mid)
        fcoords = np.array(fcoords)
        ccoords = self.fractional2cartesian(self.lattice_cell, fcoords)
        ccoords[:,0] += x_shift
        ccoords[:,1] += y_shift
        return ccoords, np.array([None for x in ccoords])

    def get_disigma_sites(self, x_shift=0.0, y_shift=0.0):
        src, dst = [], [] # src: source; dst: destination
        for i in self.one_NN_top:
            fcoord1, fcoord2 = self._top_layer_frac_coords[[i[0], i[1]],:]
            src_tmp, dst_tmp = self.shift_fcoords(fcoord1, fcoord2)
            src.append(src_tmp)
            dst.append(dst_tmp)
        src, dst = np.array(src), np.array(dst)
        src = self.fractional2cartesian(self.lattice_cell, src)
        dst = self.fractional2cartesian(self.lattice_cell, dst)
        vectors = dst - src
        src[:,0] += x_shift
        src[:,1] += y_shift
        return src, vectors

    def get_hollow_sites(self, x_shift=0.0, y_shift=0.0):
        one_NN_top = self.one_NN_top.tolist()
        num_1nn    = len(one_NN_top)
        fcoord     = []
        for i in range(num_1nn):
            for j in range(i + 1, num_1nn):
                for k in range(j + 1, num_1nn):
                    nn_set = list(set(one_NN_top[i] + one_NN_top[j] + one_NN_top[k]))
                    if len(nn_set) == 3:
                        mid = self.get_middle_fcoord(self._top_layer_frac_coords[[nn_set[0], nn_set[1], nn_set[2]],:])
                        fcoord.append(mid)
        fcoords = np.array(fcoord)
        ccoords = self.fractional2cartesian(self.lattice_cell, fcoords)
        ccoords[:,0] += x_shift
        ccoords[:,1] += y_shift
        return ccoords, np.array([None for x in ccoords])

    def fractional2cartesian(self, vector_tmp, D_coord_tmp):
        C_coord_tmp = np.dot(D_coord_tmp, vector_tmp)
        return C_coord_tmp

    def cartesian2fractional(self, vector_tmp, C_coord_tmp):
        vector_tmp = np.mat(vector_tmp)
        D_coord_tmp = np.dot(C_coord_tmp, vector_tmp.I)
        D_coord_tmp = np.array(D_coord_tmp, dtype=float)
        return D_coord_tmp

    # def get_angle(self, vector1, vector2):
    #     v1_u = vector1 / np.linalg.norm(vector1)
    #     v2_u = vector2 / np.linalg.norm(vector2)
    #     return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def write_file_with_adsorption_sites(self, adsorbate_poscar, calculation_index=0):
        site_func = dict(zip(self._support_sites, [self.get_ontop_sites, self.get_bridge_sites, self.get_hollow_sites, self.get_disigma_sites]))

        if 'C3H6_pi' == self.species:
            x_shift=0.72
            y_shift=0.0
        else:
            x_shift=0.0
            y_shift=0.0

        cart_coords, site_str, out_array=[], [], []
        for site in self.sites:
            coord_tmp, out = site_func[site](x_shift=x_shift, y_shift=y_shift)
            cart_coords.append(coord_tmp)
            site_str += [site for x in coord_tmp]
            out_array += out.tolist()
        cart_coords = np.vstack(tuple(cart_coords))

        num_sites = np.shape(cart_coords)[0]
        for site_i in range(num_sites):
            f         = StringIO(adsorbate_poscar[self.species])
            adsorbate = read(f, format='vasp')
            surface   = self._structure.copy()
            position  = (cart_coords[site_i][0], cart_coords[site_i][1])

            if site_str[site_i] == 'disigma':
                src_v = adsorbate.arrays['positions'][0] - adsorbate.arrays['positions'][1]
                src_v[2] = 0.0
                dst_v = out_array[site_i]
                dst_v[2] = 0.0
                adsorbate.rotate(src_v, dst_v, center ='COP')
                displacement = np.array([[0.0, 0.0, 0.0] for x in surface])

            add_adsorbate(surface, adsorbate, self.dist_from_surf, position)
            if site_str[site_i] == 'disigma':
                displacement_tmp = dst_v / np.linalg.norm(dst_v) * -0.72
                displacement = np.vstack([displacement, [displacement_tmp for x in range(len(adsorbate))]])
                surface.translate(displacement)
            write(f'POSCAR_{calculation_index}_{site_i}', surface, format='vasp')

adsorbate_poscar = {
'C3H6_pi':
'''C H
   1.000000000000000
 10.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000 10.0000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000 10.0000000000000000
   3   6
Selective dynamics
Direct
  0.5535920307240257  0.5054944871530235  .3586050403312200   T   T   T
  0.6352790921072941  0.3879206403028466  .4062799115126201   T   T   T
  0.4108903806304410  0.5027727193223586  .3567572395450800   T   T   T
  0.5814503326975878  0.2934868248102315  .3968419267082002   T   T   T
  0.6636640002806579  0.4018408119171915  .5113676042094202   T   T   T
  0.7319078245085368  0.3787071546929145  .3523985386757005   T   T   T
  0.6004140590434619  0.6017307633065315  .3787217106124001   T   T   T
  0.3603923215774030  0.4109270277884635  .3852183512609502   T   T   T
  0.3557746916442810  0.5933016256547884  .3802721473252602   T   T   T
''',

'C3H6_di':
'''C  H
   1.000000000000000
 10.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000 10.0000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000 10.0000000000000000
   3   6
Selective dynamics
Direct
  0.5484236650091483  0.5793696228300774  .4534433537436204   T   T   T
  0.4016582239211931  0.5674909994614906  .4441712821766108   T   T   T
  0.6238100325535113  0.4558534322141073  .4957447541409306   T   T   T
  0.5821137497425943  0.3637880141958846  .4493398337887504   T   T   T
  0.6087096830988554  0.4398812179063704  .6046439066269607   T   T   T
  0.7306517610703676  0.4604332215158591  .4783109306196105   T   T   T
  0.5860507695136543  0.6716539482048837  .4996445825796005   T   T   T
  0.3619691310064732  0.4757345003168245  .4899695206948405   T   T   T
  0.3467741410754384  0.6554008047000537  .4777358996487706   T   T   T
''',

'O':
'''O
   1.000000000000000
 10.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000 10.0000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000 10.0000000000000000
   1
Direct
0.0 0.0 0.0
''',

'OH':
'''O H
   1.000000000000000
 10.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000 10.0000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000 10.0000000000000000
   1 1
C
0.0 0.0 0.0
0.6 0.0 0.7

''',

'H':
'''H
   1.000000000000000
 10.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000 10.0000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000 10.0000000000000000
   1
C
0.0 0.0 0.0
''',

'OOH':
'''O H
   1.000000000000000
 10.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000 10.0000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000 10.0000000000000000
   2 1
C
0.0 0.0 0.0
1.290 0.0  0.733
1.290 0.985 0.733
''',

'C3H7':
'''C  H
   1.000000000000000
 10.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000 10.0000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000 10.0000000000000000
   3   7
Direct
  0.5032285934578148  0.6060872441143981  .5105109296830013
  0.4422956754542634  0.4792161991413143  .4515680270073219
  0.5080514476976327  0.3555729792148237  .5130097657332779
  0.6094002947779539  0.6164097976375560  .4841432123244549
  0.4979847519485787  0.6016567794491650  .6203520958324020
  0.4510082850485268  0.6967157765442196  .4785166738079019
  0.4960478637085886  0.3580074123423511  .6224347248530488
  0.6159067993202143  0.3518094062889375  .4926109199402289
  0.4634551722627875  0.2617368964178830  .4779179563921314
  0.3351823259455955  0.4772193591639445  .4754175078521692
'''
}

if __name__ == '__main__':
    adder = AddAtoms('../files/POSCAR_surface_example',   # input file name.
                     species='O',               # name of adsorbates. O, H, OH, OOH, C3H6_di, C3H6_pi, C3H7
                     sites='bridge',            # name of sites. ontop, bridge, hollow, disigma, or all
                     dist_from_surf=1.7,        # distance between adsorbate and the surface. Decrease this variable if the adsorbate is too high.
                     num_atomic_layer_along_Z=5) # number of atomic layers along z direction of the slab model.
    num_sites = adder.write_file_with_adsorption_sites(adsorbate_poscar)
