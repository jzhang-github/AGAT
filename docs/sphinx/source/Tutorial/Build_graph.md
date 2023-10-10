# Build graph



### Python script
You may want to build a topology graph without atomic `energy`, `forces`, `stress`, and `cell`. You can achieve this by setting `topology_only=True` when instantiating the [CrystalGraph](https://jzhang-github.github.io/AGAT/API_Docs/data/build_dataset.html#CrystalGraph) object:

```python
import os
from agat.data import CrystalGraph
from agat.data import load_graph_build_scheme

graph_build_method = load_graph_build_method(os.path.join('dataset', 'graph_build_scheme.json'))
cg = CrystalGraph(**{**graph_build_method, **{'topology_only': True}})
graph, prop = cg.get_graph('POSCAR')
print(graph.ndata)
print('==========')
print(graph.edata)
```

You can further save the graph as a binary file to the disk:
```python
from dgl import save_graphs
save_graphs("graph.bin", [graph], prop)
```
