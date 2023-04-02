# **AGAT for High-Entropy Catalysis**
### This is the manual to reproduce  results and support conclusions of [***Design High-Entropy Electrocatalyst via Interpretable Deep Graph Attention Learning***](url).   <br>    <br>
![Graphical-abstract](files/Graphical%20abstract%20-%20github.jpg)

# Table of Contents
- [Install dependencies](#install-dependencies)  
- [Example of using this code](#example-of-using-this-code)   
  - [Prepare VASP calculations](#prepare-VASP-calculations)  
  - [Collect paths of VASP calculations](#collect-paths-of-VASP-calculations)  
  - [Build graphs](#build-graphs)  
  - [Train](#train)  
  - [Predict](#predict)  
  - [High-throughput predict](#high-throughput-predict)  

- [Data for Figures](#data-and-code-for-figures) in *[reference](url)*
  - [Figure 2](#figure-2)
  - [Figure 3](#figure-3)
  - [Figure 4](#figure-4)
  - [Figure 5](#figure-5)
  - [Figure 7](#figure-7)

# Install dependencies
**Requirements file:** [requirements.txt](requirements.txt)

**Key modules**
```
dgl-cu110
numpy==1.19.5
scikit-learn==0.24.2
tensorflow==2.5.0
tensorflow-gpu==2.4.0
```
# Example of using this code
### Prepare VASP calculations
- Bulk optimization: orientation of z axis: [111]   
  The atomic positions, cell shape, and cell volume are freely relaxed.  
  
- Cleave surface: insert vacuum space along z.  
  Code: [add_vacuum_space.py](tools/add_vacuum_space.py)  
  See the [code documentation](docs/add_vacuum_space.md).  
  
- Relax the surface model.  
  Volume and shape of the surpercell is fixed.  
  
- Add adsorbate(s)  
  Code: [generate_adsorption_sites_ase.py](tools/generate_adsorption_sites_ase.py)   
  See the [code documentation](docs/generate_adsorption_sites_ase.md).  
  
- Copy generated structural file into individual folders, and run VASP.  

### Collect paths of VASP calculations   
- Find all directories containing `OUTCAR` file:   
  ```
  find . -name OUTCAR > paths.log
  ```    
- Remove the string 'OUTCAR' in `paths.log`.   
  ```
  sed -i 's/OUTCAR$//g' paths.log
  ```   

### Collect frames based on `paths.log`.  
- Code: [split_POSCAR_forces_from_vaspout_parallel.py](tools/split_POSCAR_forces_from_vaspout_parallel.py)    
- Usage: python + split_POSCAR_forces_from_vaspout_parallel.py + paths_file + dataset_path + number of cores   
- For example:  
  ``` 
  python split_POSCAR_forces_from_vaspout_parallel.py paths.log $PWD/dataset 8 # run parallelly with 8 cores.
  ```  

  - Outputs:   
  Under the `dataset` directory, four types of files are generated.
    - `fname_prop_*.csv`: `csv` files with three columns: file names of output frames, energy per atom, absolute path.   
    - `fname_prop.csv`: a `csv` file including all above files.  
    - `POSCAR_*_*_*`: `POSCAR` files seprated from VASP calculations.   
    - `POSCAR_*_*_*_force.npy`: `numpy.array` of forces.  

### Build graphs   
- Code: [Crystal2Graph.py](modules/Crystal2Graph.py)  
- Example:   
  ```
  from modules.Crystal2Graph import ReadGraphs
  import os
  if __name__ == '__main__':
      graph_reader = ReadGraphs('fname_prop.csv', # csv file generated above.
                                'dataset', # directory contain all frames
                                cutoff       = None, # We don't need this for 'ase_natural_cutoffs'.
                                mode_of_NN   = 'ase_natural_cutoffs', # identify connection between atoms with 'ase_natural_cutoffs'
                                from_binary  = False, # read from structural files
                                num_of_cores = 8, # run parallelly with 8 cores.
                                super_cell   = False # do not repeat cell for small supercells.)

      graph_list, graph_labels = graph_reader.read_all_graphs(scale_prop=False, # do not rescale the label.
                                                              ckpt_path=os.path.join('project', 'ckpt') # save the information of how to build the graphs.)
  ```   

### Train   
### Predict   
### High-throughput predict   

The reader can find the code and data of the results presented in the Figures. An example of collecting data, generating graphs, training, predicting is also provided
.
# Data and code for figures
### Figure 2
### Figure 3
### Figure 4
### Figure 5
### Figure 7

## Example

