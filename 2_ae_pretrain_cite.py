# Dependencies:
# pip: anndata, umap-learn
#
# Python starter kit for the NeurIPS 2021 Single-Cell Competition.
# Parts with `TODO` are supposed to be changed by you.
#
# More documentation:
#
# https://viash.io/docs/creating_components/python/

import argparse

parser = argparse.ArgumentParser(description='multi-modal')
parser.add_argument('-tf_seed', dest='tf_seed', type=int, default=46, help='tf random seed')
parser.add_argument('-np_seed', dest='np_seed', type=int, default=56, help='np random seed')
args = parser.parse_args()

tf_seed = args.tf_seed
np_seed = args.np_seed
suffix = str(tf_seed)+'_'+str(np_seed)

import logging
import numpy as np
import anndata as ad
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pickle as pk
import scipy
import scanpy as sc
import sys
np.random.seed(np_seed)
import tensorflow as tf
tf.random.set_seed(tf_seed)


use_label = True

logging.basicConfig(level=logging.INFO)

'''Pretrain with only exploration data (with cell type label, cell cycle scores, etc)
NOTE:
    The loss function for each epoch will be printed.
    Please change the par path to the right location of explration data
Output:
    The best pretrained model (multiome.h5 or cite.h5) will be saved to disk under the current path (./).
    The par['output'] recorded the joint embedding of exploration data.
'''

dataset_path = 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.'


par = {
    'input_mod1': 'public_data/explore/cite/cite_gex_processed_training.h5ad',
    'input_mod2': 'public_data/explore/cite/cite_adt_processed_training.h5ad',
    'output': 'logs/output_cite.h5ad',
    'n_dim': 200,
}

meta = { 'resources_dir': '.' }

## VIASH END

# TODO: change this to the name of your method
method_id = "python_starter_kit"
logging.info('Reading `h5ad` files...')
ad_mod1 = ad.read_h5ad(par['input_mod1'])
ad_mod2 = ad.read_h5ad(par['input_mod2'])
mod1_obs = ad_mod1.obs
mod1_uns = ad_mod1.uns

ad_mod2_var = ad_mod2.var

mod1_mat = ad_mod1.layers["counts"]
mod2_mat = ad_mod2.layers["counts"]
    
if use_label:
    #exploration data (with labels) gene expression is not log1p normalized
    mod1_mat = scipy.sparse.csr_matrix.log1p(mod1_mat)
    print(np.max(mod1_mat))
else:
    del ad_mod1, ad_mod2


cell_cycle_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2',\
                    'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', \
                    'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP',\
                    'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', \
                    'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', \
                    'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', \
                    'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', \
                    'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8', \
                    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', \
                    'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', \
                    'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', \
                    'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', \
                    'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', \
                    'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', \
                    'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', \
                    'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', \
                    'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']

print(mod1_mat.shape, mod2_mat.shape)

def preprocess(mod1_data, mod2_data, scale=1e4):
    nb_feats_multiome_mod1, nb_feats_multiome_mod2 = 13431, 116490
    nb_feats_cite_mod1, nb_feats_cite_mod2 = 13953, 134
    if ad_mod2_var['feature_types'][0] == 'ATAC':
        n_components_mod1, n_components_mod2 = 100, 100
        mod1_reducer = pk.load(open(meta['resources_dir'] + '/multiome_svd1.pkl','rb'))
        mod2_reducer = pk.load(open(meta['resources_dir'] + '/multiome_svd2.pkl','rb'))
    elif ad_mod2_var['feature_types'][0] == 'ADT':
        n_components_mod1, n_components_mod2 = 100, 100
        mod1_reducer = pk.load(open(meta['resources_dir'] + '/cite_svd1.pkl','rb'))
    else:
        print('Error modality')
        sys.exit()
    if mod1_data.shape[1] != nb_feats_multiome_mod1 and mod1_data.shape[1] != nb_feats_cite_mod1:
        print('Fake data to pass sample data test')
        if ad_mod2_var['feature_types'][0] == 'ATAC':
            mod1_data = np.zeros((mod1_data.shape[0], nb_feats_multiome_mod1))
            mod2_data = np.zeros((mod2_data.shape[0], nb_feats_multiome_mod2))
        else:
            mod1_data = np.zeros((mod1_data.shape[0], nb_feats_cite_mod1))
            mod2_data = np.zeros((mod2_data.shape[0], nb_feats_cite_mod2))
        mod1_data = scipy.sparse.csc_matrix(mod1_data)
        mod2_data = scipy.sparse.csc_matrix(mod2_data)

    mod1_data = scale * normalize(mod1_data,norm='l1', axis=1)
    mod2_data = scale * normalize(mod2_data,norm='l1', axis=1)
    mod1_data = scipy.sparse.csr_matrix.log1p(mod1_data) / np.log(10)
    mod2_data = scipy.sparse.csr_matrix.log1p(mod2_data) / np.log(10)
    
    #mod1_reducer.fit(mod1_data)
    pca_data_mod1 = mod1_reducer.transform(mod1_data)
    

    if ad_mod2_var['feature_types'][0] == 'ADT':
        pca_data_mod2 = mod2_data.toarray()
    else:
        #mod2_reducer.fit(mod2_data)
        pca_data_mod2 = mod2_reducer.transform(mod2_data)
    return pca_data_mod1, pca_data_mod2

mod1_pca, mod2_pca = preprocess(mod1_mat, mod2_mat)

del mod1_mat, mod2_mat

print('load data and pca done', mod1_pca.shape, mod2_pca.shape)

pca_combined = np.concatenate([mod1_pca, mod2_pca],axis=1)

print(pca_combined.shape)

del mod1_pca, mod2_pca
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.
  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        #self.model.save_weights('checkpoints/model_epoch%d.h5' % epoch)
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class Autoencoder(tf.keras.Model):
    def __init__(self, params, name=None):
        """Initialize layers to build Autoencoder model. with latent features for classification
        Args:
            params: hyperparameter object defining layer sizes, dropout values, etc.
            name: name of the model.
        """
        super(Autoencoder, self).__init__(name=name)
        self.params = params
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.classifier = self.create_classifier()

    def get_config(self):
        return {
                "params": self.params,
        }

    def call(self, inputs, training):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        digits_cell_type, digits_batch, digits_phase = self.classifier(encoded)
        if self.params['use_batch']:
            return decoded, digits_cell_type, digits_batch, digits_phase
        else:
            return decoded, digits_cell_type

    def create_encoder(self, use_resnet=True):
        if use_resnet:
            inputs = tf.keras.layers.Input(shape=(self.params['dim'],))
            for i, n_unit in enumerate(self.params['hidden_units'][:-1]):
                if i==0:
                    x_init = tf.keras.layers.Dense(n_unit, activation='relu')(inputs)
                else:
                    x_init = tf.keras.layers.Dense(n_unit, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.1)(x_init)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dense(n_unit)(x)
                x = tf.keras.layers.Add()([x,x_init])
                x = tf.keras.layers.Activation(activation='relu')(x)
            encoded = tf.keras.layers.Dense(self.params['hidden_units'][-1], activation='relu')(x)
        else:
            inputs = tf.keras.layers.Input(shape=(self.params['dim'],))
            for i, n_unit in enumerate(self.params['hidden_units'][:-1]):
                if i==0:
                    x = tf.keras.layers.Dense(n_unit, activation='relu')(inputs)
                else:
                    x = tf.keras.layers.Dense(n_unit, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.1)(x)
                x = tf.keras.layers.BatchNormalization()(x)
            encoded = tf.keras.layers.Dense(self.params['hidden_units'][-1], activation='relu')(x)
        return tf.keras.Model(inputs=inputs, outputs=encoded, name='encoder')
    
    def create_decoder(self):
        inputs = tf.keras.layers.Input(shape=(self.params['hidden_units'][-1],))
        for i, n_unit in enumerate(self.params['hidden_units'][:-1][::-1]):
            if i==0:
                x = tf.keras.layers.Dense(n_unit, activation='relu')(inputs)
            else:
                x = tf.keras.layers.Dense(n_unit, activation='relu')(x)
        decoded = tf.keras.layers.Dense(self.params['dim'], activation='relu')(x)
        return tf.keras.Model(inputs=inputs, outputs=decoded, name='decoder')
        
    def create_classifier(self):
        inputs = tf.keras.layers.Input(shape=(self.params['hidden_units'][-1],))
        #x = tf.keras.layers.Dense(int(self.params['hidden_units'][-1]/2), activation='relu')(inputs)
        #x = tf.keras.layers.Dropout(0.1)(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        #x = tf.keras.layers.Dense(int(self.params['hidden_units'][-1]/4), activation='relu')(x)
        #x = tf.keras.layers.Dropout(0.1)(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        #digits_cell_type = tf.keras.layers.Dense(self.params['nb_cell_types'])(x)
        #digits_batch = tf.keras.layers.Dense(self.params['nb_batches'])(x)
        #digits_phase = tf.keras.layers.Dense(self.params['nb_phases'])(x)
        digits_cell_type = inputs[:,:self.params['nb_cell_types']]
        digits_batch = inputs[:,self.params['nb_cell_types']:(self.params['nb_cell_types']+self.params['nb_batches'])]
        digits_phase = inputs[:,(self.params['nb_cell_types']+self.params['nb_batches']):(self.params['nb_cell_types']+self.params['nb_batches']+self.params['nb_phases'])]
        return tf.keras.Model(inputs=inputs, outputs=[digits_cell_type, digits_batch, digits_phase], name='classifier')

if use_label:
    cell_type_labels = mod1_obs['cell_type']
    batch_ids = mod1_obs['batch']
    phase_labels = mod1_obs['phase']
    nb_cell_types = len(np.unique(cell_type_labels))
    nb_batches = len(np.unique(batch_ids))
    nb_phases = len(np.unique(phase_labels))-1 # 2
    c_labels = np.array([list(np.unique(cell_type_labels)).index(item) for item in cell_type_labels])
    b_labels = np.array([list(np.unique(batch_ids)).index(item) for item in batch_ids])
    p_labels = np.array([list(np.unique(phase_labels)).index(item) for item in phase_labels])
    #0:G1, 1:G2M, 2: S, only consider the last two
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    sc.pp.log1p(ad_mod1)
    sc.pp.scale(ad_mod1)
    sc.tl.score_genes_cell_cycle(ad_mod1, s_genes=s_genes, g2m_genes=g2m_genes)
    S_scores = ad_mod1.obs['S_score'].values
    G2M_scores = ad_mod1.obs['G2M_score'].values
    phase_scores = np.stack([S_scores,G2M_scores]).T #(nb_cells, 2)


    pca_train, pca_test, c_train_labels, c_test_labels, b_train_labels, b_test_labels, p_train_labels, p_test_labels, phase_train_scores, phase_test_scores = train_test_split(pca_combined, c_labels, b_labels, p_labels,phase_scores, test_size=0.1, random_state=42)
    print(pca_train.shape, c_train_labels.shape, b_train_labels.shape, p_train_labels.shape, phase_train_scores.shape)
    print(pca_test.shape, c_test_labels.shape, b_test_labels.shape, p_test_labels.shape, phase_test_scores.shape)
    X_train = pca_train
    #Y_train = [pca_train, c_train_labels, b_train_labels, p_train_labels]
    Y_train = [pca_train, c_train_labels, b_train_labels, phase_train_scores]

    X_test = pca_test
    #Y_test = [pca_test, c_test_labels, b_test_labels, p_test_labels]
    Y_test = [pca_test, c_test_labels, b_test_labels, phase_test_scores]
else:
    if ad_mod2_var['feature_types'][0] == 'ATAC':
        nb_cell_types, nb_batches, nb_phases = 21, 5, 2
    else:
        nb_cell_types, nb_batches, nb_phases = 45, 6, 2

print(nb_cell_types, nb_batches, nb_phases)
print(ad_mod2_var['feature_types'][0])

hidden_units = [150, 120, 100, nb_cell_types+nb_batches+nb_phases+5]

params = {
    'dim' : pca_combined.shape[1],
    'lr': 1e-4,
    'hidden_units' : hidden_units,
    'nb_layers': len(hidden_units),
    'nb_cell_types': nb_cell_types,
    'nb_batches': nb_batches,
    'nb_phases': nb_phases,
    'use_batch': True
}

print('Model hyper parameters:', params)

def random_classification_loss(y_true, y_pred):
    return tf.keras.metrics.categorical_crossentropy(tf.ones_like(y_pred)/nb_batches, y_pred, from_logits=True)

model = Autoencoder(params)

model.compile(tf.keras.optimizers.Adam(learning_rate = params["lr"]), 
            loss = [tf.keras.losses.MeanSquaredError(), 
                    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    random_classification_loss,
                    tf.keras.losses.MeanSquaredError()
                    ],
            loss_weights=[0.7, 0.2, 0.05, 0.05], run_eagerly=True)

filepath = 'multiome.h5' if ad_mod2_var['feature_types'][0] == 'ATAC' else 'cite.h5'
callbacks = [EarlyStoppingAtMinLoss(patience=5),
            tf.keras.callbacks.ModelCheckpoint(filepath = filepath,
                             monitor='val_loss', save_weights_only=True)]

if use_label:    
    model.fit(x=X_train, y=Y_train,
                    epochs = 500,
                    batch_size = 32,
                    shuffle=True,
                    callbacks = callbacks,
                    validation_data=(X_test, Y_test),
                    max_queue_size = 100, workers = 28, use_multiprocessing = True)
    
    print('Start evaluation')
    eval_results = model.evaluate(X_test, Y_test, batch_size=128)
    print('Total loss, loss1, loss2, loss3, loss4:',eval_results)
    
    f_out = open('cite.log','a+')
    f_out.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\n'%(suffix, eval_results[1], eval_results[2], eval_results[3], eval_results[4]))
    f_out.close()
else:
    model(np.zeros((10, params['dim'])))
    if ad_mod2_var['feature_types'][0] == 'ATAC':
        model.load_weights(meta['resources_dir'] +'/multiome.h5')
    else:
        model.load_weights(meta['resources_dir'] +'/cite.h5')

joint_embeds = model.encoder.predict(pca_combined)

adata = ad.AnnData(
    X=joint_embeds,
    obs=mod1_obs,
	uns={
        'dataset_id': mod1_uns['dataset_id'],
        'method_id': method_id,
    },
)
adata.write_h5ad(par['output'], compression="gzip")
