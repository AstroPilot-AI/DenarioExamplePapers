The idea of this project is to combine ideas from Quantum Tensor Trains (QTT) with cosmological data, like merger trees from cosmological N-body simulations. 

We have a file containing 500 merger trees. The file is located in /Users/fvillaescusa/Documents/Software/AstroPilot/Project6/data/Pablo_merger_trees.pt

The data is stored in a PyTorch Geometric format and can be read as:

```python
import torch
f_tree = '/Users/fvillaescusa/Documents/Software/AstroPilot/Project6/data/Pablo_merger_trees.pt'
trainset = torch.load(f_tree, weights_only=False)
```

Where trainset[0] represents the first merger tree and contains the data in this format:
Data(x=[382, 4], edge_index=[2, 381], edge_attr=[381, 1], y=[1, 2], num_nodes=382, lh_id=100, mask_main=[93], node_halo_id=[382, 1])

the node features are mass, concentration, vmax, scale factor

Note that the data is not normalized, so it is better to take log and normalize before working with the data.

Please come out with an idea to combine methods from Quantum physics (e.g. QTT) and the cosmological data.

Note that PyTorch and PyTorch Geometric are already installed, but it can only run with cpus, not GPUs. Please make some plots to illustrate your findings.