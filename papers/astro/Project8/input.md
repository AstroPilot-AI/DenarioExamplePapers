We have a file containing 1000 merger trees from cosmological N-body simulations. The file is located in /Users/fvillaescusa/Documents/Software/AstroPilot/Project9/data/Pablo_merger_trees2.pt

The data is stored in PyTorch Geometric format, and can be read as:

```python
import torch
f_tree = '/Users/fvillaescusa/Documents/Software/AstroPilot/Project9/data/Pablo_merger_trees2.pt'
trainset = torch.load(f_tree, weights_only=False)
```

Where trainset[0] represents the first merger tree and contains the data in this format:
Data(x=[382, 4], edge_index=[2, 381], edge_attr=[381, 1], y=[1, 2], num_nodes=382, lh_id=100, mask_main=[93], node_halo_id=[382, 1])

x represents the node features. y is the value of the cosmological parameters, Omega_m and sigma_8. The node features are mass, concentration, vmax, scale factor. Note that the data is not normalized, so it is better to take log and normalize before working with the data.

The scale factor ranges from 0 (beginning of the universe) to 1 (current time). Each node represents a dark matter halo and is characterized by the four values mentioned above: halo mass, halo concentration, halo Vmax (maximum circular velocity), and scale factor.

Please come out with an idea to to explore this data using state-of-the-art geometric deep learning techniques.

Note that PyTorch and PyTorch Geometric are already installed, but it can only run with cpus, not GPUs. Please make some plots to illustrate your findings.