import scanpy as sc
import numpy as np
def preprocess(data):
    adata = sc.AnnData(data)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)


    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_high = adata[:, adata.var.highly_variable]
    return adata.X