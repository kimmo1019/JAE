import numpy as np
import scipy.sparse as sp
import anndata as ad
from sklearn.decomposition import TruncatedSVD
import pickle as pk
from sklearn.preprocessing import normalize

'''SVD applied to all available data (aggregate all) for multiome and cite
NOTE:
    For cite-seq, phase1 is a strict subset of phase2.
    For multiome, phase1 contains 1511 cells that is not in phase2. We intent to merge them.
    The raw phase1 and phase2 data is under the DATA_DIR folder, respectively.
    Please change DATA_DIR to the right folder.
Output:
    Two SVDs for multiome mod1 and mod2, and SVD for cite mod1 will be saved to disk under the current path (./).
'''

# TODO: change this to the directory of your own
DATA_DIR = 'aggragate_data'

phase1_multiome_mod1 = ad.read_h5ad("%s/phase1/joint_embedding/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.censor_dataset.output_mod1.h5ad"%DATA_DIR)
phase1_multiome_mod2 = ad.read_h5ad("%s/phase1/joint_embedding/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.censor_dataset.output_mod2.h5ad"%DATA_DIR)
phase1_cite_mod1 = ad.read_h5ad("%s/phase1/joint_embedding/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.censor_dataset.output_mod1.h5ad"%DATA_DIR)
phase1_cite_mod2 = ad.read_h5ad("%s/phase1/joint_embedding/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.censor_dataset.output_mod2.h5ad"%DATA_DIR)
print(phase1_multiome_mod1.shape, phase1_multiome_mod2.shape)
print(phase1_cite_mod1.shape, phase1_cite_mod2.shape)
#(22463, 13431) (22463, 116490)
#(43890, 13953) (43890, 134)


phase2_multiome_mod1 = ad.read_h5ad("%s/phase2/joint_embedding/openproblems_bmmc_multiome_phase2/openproblems_bmmc_multiome_phase2.censor_dataset.output_mod1.h5ad"%DATA_DIR)
phase2_multiome_mod2 = ad.read_h5ad("%s/phase2/joint_embedding/openproblems_bmmc_multiome_phase2/openproblems_bmmc_multiome_phase2.censor_dataset.output_mod2.h5ad"%DATA_DIR)
phase2_cite_mod1 = ad.read_h5ad("%s/phase2/joint_embedding/openproblems_bmmc_cite_phase2/openproblems_bmmc_cite_phase2.censor_dataset.output_mod1.h5ad"%DATA_DIR)
phase2_cite_mod2 = ad.read_h5ad("%s/phase2/joint_embedding/openproblems_bmmc_cite_phase2/openproblems_bmmc_cite_phase2.censor_dataset.output_mod2.h5ad"%DATA_DIR)
print(phase2_multiome_mod1.shape, phase2_multiome_mod2.shape)
print(phase2_cite_mod1.shape, phase2_cite_mod2.shape)
#(42492, 13431) (42492, 116490)
#(66175, 13953) (66175, 134)

phase1_multiome_cells = list(phase1_multiome_mod1.obs.index)
phase2_multiome_cells = list(phase2_multiome_mod1.obs.index)

#merge phase1 and phase2 multiome data
idx = [i for i,item in enumerate(phase2_multiome_cells) if item not in phase1_multiome_cells]
agg_multiome_mod1_count = sp.vstack([phase1_multiome_mod1.layers["counts"], phase2_multiome_mod1.layers["counts"][idx, :]])
agg_multiome_mod2_count = sp.vstack([phase1_multiome_mod2.layers["counts"], phase2_multiome_mod2.layers["counts"][idx, :]])

#scale and log transform
random_seed = 123
scale = 1e4
mod1_data = scale * normalize(agg_multiome_mod1_count,norm='l1', axis=1)
mod1_data = sp.csr_matrix.log1p(mod1_data) / np.log(10)

mod2_data = scale * normalize(agg_multiome_mod2_count,norm='l1', axis=1)
mod2_data = sp.csr_matrix.log1p(mod2_data) / np.log(10)

#apply SVD to both modalities of multiome
for n_components_mod1, n_components_mod2 in [(100,100)]:
    mod1_reducer = TruncatedSVD(n_components=n_components_mod1, random_state=random_seed)
    mod1_reducer.fit(mod1_data)
    pca_data_mod1 = mod1_reducer.transform(mod1_data)
    #print('multiome 1 done',pca_data_mod1.shape)
    pk.dump(mod1_reducer, open("multiome_svd1.pkl","wb"))

    mod2_reducer = TruncatedSVD(n_components=n_components_mod2, random_state=random_seed)
    mod2_reducer.fit(mod2_data)
    pca_data_mod2 = mod2_reducer.transform(mod2_data)
    #print('multiome 2 done',pca_data_mod2.shape)
    pk.dump(mod2_reducer, open("multiome_svd2.pkl","wb"))


#apply SVD only to GEX of cite
agg_cite_mod1_count = phase2_cite_mod1.layers['counts']
mod1_data = scale * normalize(agg_cite_mod1_count,norm='l1', axis=1)
mod1_data = sp.csr_matrix.log1p(mod1_data) / np.log(10)

for n_components_mod1 in [100]:
    mod1_reducer = TruncatedSVD(n_components=n_components_mod1, random_state=random_seed)
    mod1_reducer.fit(mod1_data)
    pca_data_mod1 = mod1_reducer.transform(mod1_data)
    #print('cite 1 done',pca_data_mod1.shape)
    pk.dump(mod1_reducer, open("cite_svd1.pkl","wb"))
