# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:30:07 2021

@author: ZHANG Jun
"""
#
from ase.io import read, write
import sys
import os
import numpy as np
import multiprocessing
# from pymatgen.io.vasp.outputs import Outcar

# VASP calculates forces using 'free energy': https://www.vasp.at/wiki/index.php/Forces

def read_oszicar(fname='OSZICAR'):
    ee_steps = []
    with open(fname, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'E0=' in line.split():
            ee_steps.append(int(lines[i-1].split()[1]))
    return ee_steps

def read_incar(fname='INCAR'):
    with open(fname, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'NELM' in line.split():
            NELM = int(line.split()[2])
            break
    return NELM

def split_output(in_path_list, out_path, working_dir, process_index, mask_similar_frames=True, energy_stride=0.05):
    '''
    :param in_path_list: A list of absolute paths where OUTCAR and XDATCAR files exist.
    :type in_path_list: list
    :param out_path: Absolute path where the collected data to save.
    :type out_path: str
    :param working_dir: The working directory.
    :type working_dir: str
    '''

    print('mask_similar_frames', mask_similar_frames)
    f_csv = open(os.path.join(out_path, f'fname_prop_{process_index}.csv'), 'w', buffering=1)
    for path_index, in_path in enumerate(in_path_list):
        in_path = in_path.strip("\n")
        os.chdir(in_path)

        if os.path.exists('OUTCAR') and os.path.exists('XDATCAR') and os.path.exists('OSZICAR') and os.path.exists('INCAR') and os.path.exists('CONTCAR'):

            # read free energy
            with open('OUTCAR', 'r') as f:
                free_energy = []
                for line in f:
                    if '  free  energy   TOTEN  ' in line.split('='):
                        free_energy.append(float(line.split()[4]))

            # if len(free_energy) > 1:
            read_good = True
            try:
                # read frames
                frame_contcar  = read('CONTCAR')
                constraints    = frame_contcar.constraints
                frames_outcar  = read('OUTCAR', index=':')   # coordinates in OUTCAR file are less accurate than that in XDATCAR. Energy in OUTCAR file is more accurate than that in OSZICAR file
                frames_xdatcar = read('XDATCAR', index=':')
                [x.set_constraint(constraints) for x in frames_xdatcar]

                num_atoms      = len(frames_outcar[0])
                num_frames     = len(frames_outcar)
                assert len(frames_outcar) == len(frames_xdatcar), f'Inconsistent number of frames between OUTCAR and XDATCAR files. OUTCAR: {len(frames_outcar)}; XDATCAR {len(frames_xdatcar)}'
                assert num_frames == len(free_energy), 'Number of frams does not equal to number of free energies.'
                ee_steps = read_oszicar()
                assert len(ee_steps) == len(frames_xdatcar), f'Inconsistent number of frames between OSZICAR and XDATCAR files. OSZICAR: {len(ee_steps)}; XDATCAR {len(frames_xdatcar)}'
                NELM     = read_incar()
            except:
                print(f'Read OUTCAR, OSZICAR, INCAR, CONTCAR, and/or XDATCAR with exception in: {in_path}')
                read_good = False

            if read_good:
                ###############################################################
                # mask similar frames based on energy difference.
                if mask_similar_frames:
                    no_mask_list = [0]
                    for i, e in enumerate(free_energy):
                        if abs(e - free_energy[no_mask_list[-1]]) > energy_stride:
                            no_mask_list.append(i)
                    if not no_mask_list[-1] == num_frames - 1:
                        no_mask_list.append(num_frames - 1)

                    free_energy    = [free_energy[x] for x in no_mask_list]
                    frames_outcar  = [frames_outcar[x] for x in no_mask_list]
                    frames_xdatcar = [frames_xdatcar[x] for x in no_mask_list]
                    ee_steps       = [ee_steps[x] for x in no_mask_list]
                else:
                    no_mask_list = [x for x in range(num_frames)]
                ###############################################################

                free_energy_per_atom = [x / num_atoms for x in free_energy]

                # save frames
                for i in range(len(no_mask_list)):
                    if ee_steps[i] < NELM:
                        fname = str(os.path.join(out_path, f'POSCAR_{process_index}_{path_index}_{i}'))
                        while os.path.exists(fname):
                            fname = fname + '_new'

                        frames_xdatcar[i].write(fname)
                        forces = frames_outcar[i].get_forces(apply_constraint=False)
                        np.save(fname + '_force.npy', forces)
                        # print(os.path.basename(fname) + ',  ')
                        # print(str(free_energy_per_atom[i]) + ',  ' + str(in_path))
                        f_csv.write(os.path.basename(fname) + ',  ')
                        f_csv.write(str(free_energy_per_atom[i]) + ',  ' + str(in_path) + '\n')
                    else:
                        print(f'Electronic steps of {i}th ionic step greater than NELM: {NELM} in path: {in_path}')

        else:
            print(f'OUTCAR, OSZICAR, INCAR, CONTCAR, and/or XDATCAR files do not exist in {in_path}.')

        os.chdir(working_dir)
    f_csv.close()
    return None

if __name__ == '__main__':
    assert len(sys.argv) > 3, 'Usage: command + paths_file + dataset_path + number of cores'
    paths_file      = sys.argv[1] # a file contains many paths
    dataset_path    = sys.argv[2] #'./dataset'
    num_cores       = int(sys.argv[3])
    CURDIR          = os.getcwd()
    mask_similar_frames = False
    energy_stride   = 0.1
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    with open(paths_file, 'r') as f:
        paths = []
        for i in f:
            paths.append(i.strip())

    # num_paths_per_core = len(paths) // num_cores + 1
    # line_index         = [x for x in range(len(paths))]
    # batch_index        = [line_index[x: x + num_paths_per_core] for x in range(0, len(paths), num_paths_per_core)]

    batch_index = np.array_split([x for x in range(len(paths))], num_cores)

    processes = []
    for process_index, path_index in enumerate(batch_index):
        path_list = [paths[x] for x in path_index]
        p = multiprocessing.Process(target=split_output, args=[path_list, dataset_path, CURDIR, process_index, mask_similar_frames, energy_stride])
        p.start()
        processes.append(p)
    print(processes)

    for process in processes:
        process.join()

    f = open(os.path.join(dataset_path, 'fname_prop.csv'), 'w')

    for job, _ in enumerate(batch_index):
        lines = np.loadtxt(os.path.join(dataset_path, f'fname_prop_{job}.csv'),dtype=str)
        np.savetxt(f, lines, fmt='%s')

    f.close()
