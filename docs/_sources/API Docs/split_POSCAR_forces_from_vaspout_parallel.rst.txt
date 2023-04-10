################################################
split_POSCAR_forces_from_vaspout_parallel.py
################################################


.. Note:: Example of using this code: `AGAT_CATA_example`_

.. Important:: Outputs of this script is stored at ``out_path``.

.. function:: read_oszicar(fname='OSZICAR')

   Read `OSZICAR`_ file.
   
   :param str fname: 'OSZICAR' or another file name with `OSZICAR`_ format.
   :Returns: a list of electronic steps of each ionic steps.
   :rtype: list
   

.. function:: read_incar(fname='INCAR')

   Read `INCAR`_ file.

   :param str fname: 'INCAR' or another file name with `INCAR`_ format.


.. function:: split_output(in_path_list, out_path, working_dir, process_index, mask_similar_frames=True, energy_stride=0.05)

   Read `VASP`_ outputs and seperate the outputs into a dataset.


   .. Note:: You may find information at `AGAT_CATA_example`_.


   :param in_path_list: A list of absolute paths where OUTCAR and XDATCAR files exist.
   :type in_path_list: list
   :param out_path: Absolute path where the collected data to save.
   :type out_path: str
   :param working_dir: The working directory.
   :type working_dir: str
   :param int/str process_index: index to differentiate output file names.
   :param mask_similar_frames bool: if True, mask frames in `VASP`_ `OUTCAR`_ with similar energies.
   
      .. Note:: If ``mask_similar_frames=True``, the energy difference between collected frames is larger than ``energy_stride`` defined below.
      
   :param float energy_stride: specify energy increaments if you need to mask similar frames.
   

.. method:: __main__

   Main process to seperate `VASP`_ outputs in parallel.







.. _AGAT_CATA_example: https://github.com/jzhang-github/AGAT/tree/main/AGAT_CATA#example-of-using-this-code
.. _OSZICAR: https://www.vasp.at/wiki/index.php/OSZICAR
.. _INCAR: https://www.vasp.at/wiki/index.php/INCAR
.. _VASP: https://www.vasp.at/
.. _OUTCAR: https://www.vasp.at/wiki/index.php/OUTCAR
