# JAE
Single cell joint embedding and modality prediction with autoencoder. JAE achieves the first place in [NeurIPS 2021 single cell competition](https://openproblems.bio/neurips_2021/) Joint Embedding tasks (see [Leaderboard](https://eval.ai/web/challenges/challenge-page/1111/leaderboard/2863),team: Amateur). This work was invited to give a talk at NeurIPS 2021 workshop (see slides here).

<img src="model_architecture.png" width="70%">

## Model description

In brief, we built an autoencoder for joint embedding (JAE). Each modality will first be SVD transformed and concatenated together (denoted as x). The major difference from standard AE is that we incorporated the information from cell annotations (e.g., cell label, cell cycle score, and cell batch) to constrain the structure of latent features. We desire that some latent features (c) predict the cell type information, some features predict the cell cycle score. Noticeably, for feature (b), we want it to predict the batch label as randomly as possible to potentially eliminate the batch effect. z has no constrain at all to ensure the flexibility of neural network.

In the pretrain stage, JAE was trained with exploration data where the cell annotation information (cell type, cell cycle phase score) is available. In the test stage where the cell annotation information is not available, we only minimize the reconstruction loss of the autoencoder with a smaller learning rate (fine-tune).

## Environment

- Python == 3.9.0
- TensorFlow == 2.7.0
- cuda == 11.2.0
- cudnn == 8.1.1.33

## Model pretrain

**Step 1**: SVD transformation to each modality (except the ADT) for dimension reduction

One can run `python 1_pca_pretrain.py` to get the SVD transformation for each modality. Note that this SVD transformation was applied to aggregated dataset where only part of the cells were annotated.

**Step 2**: Autoencoder pretrain with regularizing latent features simulteneously

One can run `python 2_ae_pretrain_multiome.py` or `python 2_ae_pretrain_cite.py` to pretrain JAE with Multiome or CITE-seq data, respectively. The pretrained model weight file will be saved.



