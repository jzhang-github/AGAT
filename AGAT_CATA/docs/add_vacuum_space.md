# Manual for [add_vacuum_space.py](https://github.com/jzhang-github/AGAT/blob/main/AGAT_CATA/tools/add_vacuum_space.py)  

* Structural file: [CONTCAR_bulk_example](CONTCAR_bulk_example)  

* Run:  
```  
python add_vacuum_space.py CONTCAR_bulk_example 6  
```  

* Output file: [CONTCAR_bulk_example_with_vacuum](CONTCAR_bulk_example_with_vacuum)

* Note:  
	* There are 6 atomic planes along z in [CONTCAR_bulk_example](CONTCAR_bulk_example) file.
	* Atoms in the bottom plane will be removed by this code.  
	* Thickness of vacuum space: 10 Å.  
	* Bottom two atomic layers are fixed to their bulk positions.
