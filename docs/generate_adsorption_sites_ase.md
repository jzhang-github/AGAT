# Manual for [generate_adsorption_sites_ase.py](../tools/add_vacuum_space.py)  

* Usage: python generate_adsorption_sites_ase.py  

* Structural file: [POSCAR_surface_example](../files/POSCAR_surface_example)  

* Modify these [lines](../tools/generate_adsorption_sites_ase.py#L331-L335).  

* Change working directory to [AGAT/AGAT_CATA](../)  
  
* Run:  
```  
python tools\generate_adsorption_sites_ase.py    
```  

* Output files: `POSCAR_0_[0-47]`. 48 POSCAR files are generated under the current directory. Here is an example: [POSCAR_0_0](../files/POSCAR_0_0)

# API documentation of [`AddAtoms`](../tools/generate_adsorption_sites_ase.py#L15) object in [generate_adsorption_sites_ase.py](../tools/generate_adsorption_sites_ase.py)  


```  
class AddAtoms(fname, sites='all', species='O', num_atomic_layer_along_Z=3, dist_from_surf=2.2, crystal_type='fcc')
```  
Place adsorbate on a clean surface.  
- Parameters: 
	- fname (*str*): file name of the input structure.  
	
	- sites (*str*/*list*): name of the adsorption site(s).   
	  If it is a string, it can be 'all', 'ontop', 'bridge', or 'hollow'. If it is 'all', the `sites` will be include all [support_sites](../tools/generate_adsorption_sites_ase.py#L21).   
	  It can also be a *list*. For example: `['ontop', 'bridge', 'hollow']`.  
	 
	  
	- species (*str*/*list*): name of the adsorbate.  
	  This argument can be 'O', 'H', 'OH', or 'OOH'.  
          You can define more adsorbates like [these lines](../tools/generate_adsorption_sites_ase.py#L220-L328).  
	  
	- dist_from_surf (*float*): Distance of the adsorbate from the surface.  
	Decrease this variable if the adsorbate is too high.    
	
	- num_atomic_layer_along_Z: (*int*): Number of atomic layers along z direction.  

- Method:
	- `write_file_with_adsorption_sites(adsorbate_poscar, calculation_index=0)`
	- Parameters:
		- adsorbate_poscar (*dict*): A Python dictionary for the structural information of adsorbates. For example: [`adsorbate_poscar`](../tools/generate_adsorption_sites_ase.py#L220-L328)    
		- calculation_index (*str*/*int*): For the convenience of output file names.   

**Example:**

```
if __name__ == '__main__':
    adder = AddAtoms('POSCAR_surface_example',
                     species='O',
                     sites='bridge',
                     dist_from_surf=1.7,
                     num_atomic_layer_along_Z=5)
    num_sites = adder.write_file_with_adsorption_sites(adsorbate_poscar)
 ```  
 Adapted from [here](../tools/generate_adsorption_sites_ase.py#L330-L336).  
