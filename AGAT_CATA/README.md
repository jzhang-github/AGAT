# **AGAT for High-Entropy Catalysis**
### This is the manual to reproduce  results and support conclusions of [***Design High-Entropy Electrocatalyst via Interpretable Deep Graph Attention Learning***](url).   <br>    <br>
![Graphical-abstract](files/Graphical%20abstract%20-%20github.jpg)

# Table of Contents
- [Dependencies](#dependencies)  
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

# Dependencies
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
  See the [Manual](docs/add_vacuum_space.md).  
  
- Relax the surface model.  
  Volume and shape of the surpercell is fixed.  
  
- Add adsorbate(s)  
  Code: [generate_adsorption_sites_ase.py](tools/generate_adsorption_sites_ase.py)
  See the [Manual](docs/generate_adsorption_sites_ase.md).  
  
### Collect paths of VASP calculations
### Build graphs
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

