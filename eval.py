#joint embedding evaluation give embeddings with six metrics

import scib
import anndata as ad
import scanpy as sc
import numpy as np


def load_data(path='top_method'):
    '''adata: the predicted embeddings
       adata_sol: the provided solution
    '''
    adata = ad.read_h5ad('%s/output_cite.h5ad'%path) 
    adata_sol = ad.read_h5ad('%s/openproblems_bmmc_cite_phase2.censor_dataset.output_solution.h5ad'%path) 
    adata.obs['batch'] = adata_sol.obs['batch'][adata.obs_names]
    adata.obs['cell_type'] = adata_sol.obs['cell_type'][adata.obs_names]
    print(adata.shape,adata_sol.shape)
    adata_bc = adata.obs_names
    adata_sol_bc = adata_sol.obs_names
    select = [item in adata_bc for item in adata_sol_bc]
    adata_sol = adata_sol[select, :]
    print(adata.shape, adata_sol.shape)
    return adata, adata_sol

def get_nmi(adata):
    print('Preprocessing')
    sc.pp.neighbors(adata, use_rep='X_emb')
    print('Clustering')
    scib.cl.opt_louvain(
        adata,
        label_key='cell_type',
        cluster_key='cluster',
        plot=False,
        inplace=True,
        force=True
    )
    print('Compute score')
    score = scib.me.nmi(adata, group1='cluster', group2='cell_type')
    return score

def get_cell_type_ASW(adata):
    return scib.me.silhouette(adata, group_key='cell_type', embed='X_emb')

def get_cell_cycle_conservation(adata, adata_solution):
    recompute_cc = 'S_score' not in adata_solution.obs_keys() or \
            'G2M_score' not in adata_solution.obs_keys()
    organism = adata_solution.uns['organism']
    print('Compute score')
    score = scib.me.cell_cycle(
        adata_pre=adata_solution,
        adata_post=adata,
        batch_key='batch',
        embed='X_emb',
        recompute_cc=recompute_cc,
        organism=organism
    )
    return score

def get_traj_conservation(adata, adata_solution):
    adt_atac_trajectory = 'pseudotime_order_ATAC' if 'pseudotime_order_ATAC' in adata_solution.obs else 'pseudotime_order_ADT'
    sc.pp.neighbors(adata, use_rep='X_emb')
    obs_keys = adata_solution.obs_keys()
    if 'pseudotime_order_GEX' in obs_keys:
        score_rna = scib.me.trajectory_conservation(
            adata_pre=adata_solution,
            adata_post=adata,
            label_key='cell_type',
            pseudotime_key='pseudotime_order_GEX'
        )
    else:
        score_rna = np.nan

    if adt_atac_trajectory in obs_keys:
        score_adt_atac = scib.me.trajectory_conservation(
            adata_pre=adata_solution,
            adata_post=adata,
            label_key='cell_type',
            pseudotime_key=adt_atac_trajectory
        )
    else:
        score_adt_atac = np.nan

    score_mean = (score_rna + score_adt_atac) / 2
    return score_mean

def get_batch_ASW(adata):
    score = scib.me.silhouette_batch(
        adata,
        batch_key='batch',
        group_key='cell_type',
        embed='X_emb',
        verbose=False
    )
    return score

def get_graph_connectivity(adata):
    sc.pp.neighbors(adata, use_rep='X_emb')
    print('Compute score')
    score = scib.me.graph_connectivity(adata, label_key='cell_type')
    return score

if __name__ == "__main__":
    adata, adata_sol = load_data()
    adata.obsm['X_emb'] = adata.X
    #print(adata.obs['cell_type'],adata.obs['batch'] )
    nmi = get_nmi(adata)
    cell_type_asw = get_cell_type_ASW(adata)
    cc_con = get_cell_cycle_conservation(adata, adata_sol)
    traj_con = get_traj_conservation(adata, adata_sol)
    batch_asw = get_batch_ASW(adata)
    graph_score = get_graph_connectivity(adata)
    print(nmi, cell_type_asw, cc_con, traj_con, batch_asw, graph_score)
    print('average metric: %.5f'%np.mean([nmi, cell_type_asw, cc_con, traj_con, batch_asw, graph_score]))
