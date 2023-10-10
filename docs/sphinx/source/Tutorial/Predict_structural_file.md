# Build graph



### Python script
To predict a structure file, you can use:

```python
import os
import torch
from agat.data import CrystalGraph
from agat.data import load_graph_build_scheme
from agat.lib import load_model

# load graph from structure file
graph_build_method = load_graph_build_scheme(os.path.join('agat_model', 'graph_build_scheme.json'))
cg = CrystalGraph(**{**graph_build_method, **{'topology_only': True}})
graph, prop = cg.get_graph('POSCAR')
graph = graph.to('cuda')

# load the model
model = load_model('agat_model', device='cuda')
with torch.no_grad():
    energy_per_atom, force, stress = model(graph)
```

Before using this script, you need to prepare:
- a `POSCAR` file;
- a `graph_build_scheme.json` file, which is normally saved when you build your database;
- a well-trained model: `agat.pth`.