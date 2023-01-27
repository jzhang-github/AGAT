# Distributed under the terms of the MIT License.

__author__ = "Shyue Ping Ong, Geoffroy Hautier, Sai Jayaraman"
__copyright__ = "Copyright 2011, The Materials Project"

# =============================================================================
# Original author: "Shyue Ping Ong, Geoffroy Hautier, Sai Jayaraman"
# ZHANG Jun modified this script on 2021/05/14 (yyyy/mm/dd).
# Detail: get_connections() are replaced by get_connections_new(). The new one
# has a faster speed and more suitable for constructing a DGL graph.
# =============================================================================

from warnings import warn
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.core.sites import PeriodicSite
from math import acos, pi
from pymatgen.util.num import abs_cap

class VoronoiConnectivity:
    """
    Computes the solid angles swept out by the shared face of the voronoi
    polyhedron between two sites.
    """

    def __init__(self, structure, cutoff=10):
        """
        Args:
            structure (Structure): Input structure
            cutoff (float) Cutoff distance.
        """
        self.cutoff = cutoff
        self.s = structure
        recp_len = np.array(self.s.lattice.reciprocal_lattice.abc)
        i = np.ceil(cutoff * recp_len / (2 * pi))
        offsets = np.mgrid[-i[0] : i[0] + 1, -i[1] : i[1] + 1, -i[2] : i[2] + 1].T
        self.offsets = np.reshape(offsets, (-1, 3))
        # shape = [image, axis]
        self.cart_offsets = self.s.lattice.get_cartesian_coords(self.offsets)

    @property
    def connectivity_array(self):
        """
        Provides connectivity array.

        Returns:
            connectivity: An array of shape [atomi, atomj, imagej]. atomi is
            the index of the atom in the input structure. Since the second
            atom can be outside of the unit cell, it must be described
            by both an atom index and an image index. Array data is the
            solid angle of polygon between atomi and imagej of atomj
        """
        # shape = [site, axis]
        cart_coords = np.array(self.s.cart_coords)
        # shape = [site, image, axis]
        all_sites = cart_coords[:, None, :] + self.cart_offsets[None, :, :]
        vt = Voronoi(all_sites.reshape((-1, 3)))
        n_images = all_sites.shape[1]
        cs = (len(self.s), len(self.s), len(self.cart_offsets))
        connectivity = np.zeros(cs)
        vts = np.array(vt.vertices)
        for (ki, kj), v in vt.ridge_dict.items():
            atomi = ki // n_images
            atomj = kj // n_images

            imagei = ki % n_images
            imagej = kj % n_images

            if imagei != n_images // 2 and imagej != n_images // 2:
                continue

            if imagei == n_images // 2:
                # atomi is in original cell
                val = solid_angle(vt.points[ki], vts[v])
                connectivity[atomi, atomj, imagej] = val

            if imagej == n_images // 2:
                # atomj is in original cell
                val = solid_angle(vt.points[kj], vts[v])
                connectivity[atomj, atomi, imagei] = val

            if -10.101 in vts[v]:
                warn("Found connectivity with infinite vertex. " "Cutoff is too low, and results may be " "incorrect")
        return connectivity

    @property
    def max_connectivity(self):
        """
        returns the 2d array [sitei, sitej] that represents
        the maximum connectivity of site i to any periodic
        image of site j
        """
        return np.max(self.connectivity_array, axis=2)

    def get_connections_new(self):     # modified by ZHANG Jun on 20210421, to build dgl graph
        """
        Returns a list of site pairs that are Voronoi Neighbors, along
        with their real-space distances.

        After test, this function returns the correct connections for a
        supercell big enough.
        """
        sender, receiver, dist = [], [], []
        maxconn = self.max_connectivity
        for ii in range(0, maxconn.shape[0]):
            for jj in range(ii + 1, maxconn.shape[1]):
                if maxconn[ii][jj] != 0:
                    dist.append(self.s.get_distance(ii, jj))
                    sender.append(ii)
                    receiver.append(jj)
        # print(np.shape(sender), np.shape(receiver), np.shape(dist))
        bsender   = sender   + receiver      # bidirectional
        breceiver = receiver + sender        # bidirectional
        bsender  += [x for x in range(maxconn.shape[0])] # add self_loop
        breceiver+= [x for x in range(maxconn.shape[0])] # add self_loop
        dist     *= 2                        # bidirectional
        dist     += [0.0] * maxconn.shape[0] # add self loop for `dist`
        # print(np.shape(bsender), np.shape(breceiver), np.shape(dist))
        dist      = [[x] for x in dist]
        # https://docs.dgl.ai/guide_cn/graph-feature.html 通过张量分配创建特征时，DGL会将特征赋给图中的每个节点和每条边。该张量的第一维必须与图中节点或边的数量一致。 不能将特征赋给图中节点或边的子集。
        return bsender, breceiver, dist

    def get_sitej(self, site_index, image_index):
        """
        Assuming there is some value in the connectivity array at indices
        (1, 3, 12). sitei can be obtained directly from the input structure
        (structure[1]). sitej can be obtained by passing 3, 12 to this function

        Args:
            site_index (int): index of the site (3 in the example)
            image_index (int): index of the image (12 in the example)
        """
        atoms_n_occu = self.s[site_index].species
        lattice = self.s.lattice
        coords = self.s[site_index].frac_coords + self.offsets[image_index]
        return PeriodicSite(atoms_n_occu, coords, lattice)

def solid_angle(center, coords):
    """
    Helper method to calculate the solid angle of a set of coords from the
    center.
    Args:
        center (3x1 array): Center to measure solid angle from.
        coords (Nx3 array): List of coords to determine solid angle.
    Returns:
        The solid angle.
    """
    o = np.array(center)
    r = [np.array(c) - o for c in coords]
    r.append(r[0])
    n = [np.cross(r[i + 1], r[i]) for i in range(len(r) - 1)]
    n.append(np.cross(r[1], r[0]))
    vals = []
    for i in range(len(n) - 1):
        v = -np.dot(n[i], n[i + 1]) / (np.linalg.norm(n[i]) * np.linalg.norm(n[i + 1]))
        vals.append(acos(abs_cap(v)))
    phi = sum(vals)
    return phi + (3 - len(r)) * pi
