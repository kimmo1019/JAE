# JAE
Single cell joint embedding and modality prediction with autoencoder. JAE achieves the first place in [NeurIPS 2021 single cell competition](https://openproblems.bio/neurips_2021/) Joint Embedding tasks (see [Leaderboard](https://eval.ai/web/challenges/challenge-page/1111/leaderboard/2863)).

<img src="model_architecture.png" width="70%">

## Model description

In brief, we built an autoencoder for joint embedding (JAE). Each modality will first be SVD transformed and concatenated together (denoted as x). The major difference from standard AE is that we incorporated the information from cell annotations (e.g., cell label, cell cycle score, and cell batch) to constrain the structure of latent features. We desire that some latent features (c) predict the cell type information, some features predict the cell cycle score. Noticeably, for feature (b), we want it to predict the batch label as randomly as possible to potentially eliminate the batch effect. z has no constrain at all to ensure the flexibility of neural network.

In the pretrain stage, JAE was trained with exploration data where the cell annotation information (cell type, cell cycle phase score) is available. In the test stage where the cell annotation information is not available, we only minimize the reconstruction loss of the autoencoder with a smaller learning rate (fine-tune).

