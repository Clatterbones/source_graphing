# Thesis at University of Copenhagen, 2025

This project was made for personal exploratory use rather than generalizability. But roughly, the notebooks in `nbs` should be run in numerical order.
- `1_seed_dataset_construction.ipynb` imports, explores and wrangles data into a single dataset.
- `2_graph_population.ipynb` connects the data to the WikiNews knowledge graph to create a graphical dataset.
- `3_gnn.ipynb` applies to it a PyTorch_Geometric graphical neural network, pushes source reliability, and contains functionality for extracting model prediction visualization subgraphs using GNNExplainer.

`sparqlgenerator.py` and `utility.py` in `src` are smaller libraries. The former generally contains string functions for generating SPARQL queries as a function of a dataset. The latter contains miscellaneous functions for handling the datasets agnostically and handling the downstream graphical objects.
