import os
import sys
import time
import numpy as np
from glob import glob

import logging
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du
import vae_utils as vu
import model_utils as mu

def precip_norm(x):
    return np.log(x+1)

# ======================= Hyperparameters ======================= #

filter_nums = [64, 128] # number of convolution kernels per down-/upsampling layer 
latent_dim = 8 # number of latent feature channels
activation = 'relu' # activation function
num_embeddings = 128 # number of the VQ codes

input_size = (128, 256, 1) # size of MRMS input
latent_size = (32, 64, latent_dim) # size of compressed latent features

load_weights = True

# location of the previous weights
model_name_load = '/glade/work/ksha/GAN/models/VQ_VAE_025_{}_{}_L{}_N{}_{}_base'.format(
    filter_nums[0], filter_nums[1], latent_dim, num_embeddings, activation)
# location for saving new weights
model_name_save = '/glade/work/ksha/GAN/models/VQ_VAE_025_{}_{}_L{}_N{}_{}_base'.format(
    filter_nums[0], filter_nums[1], latent_dim, num_embeddings, activation)

lr = 1e-4 # learning rate
# samples per epoch = N_batch * batch_size
epochs = 99999
N_batch = 64
batch_size = 64

# ====================== Model design ===================== #

# ---------------- encoder ----------------- #

encoder_in = keras.Input(shape=input_size)
X = encoder_in

X = layers.Conv2D(filter_nums[0], 3, padding="same")(X)
X = layers.BatchNormalization()(X)
X = layers.Activation(activation)(X)

X = layers.Conv2D(filter_nums[0], 3, strides=2, padding="same")(X)
X = layers.BatchNormalization()(X)
X = layers.Activation(activation)(X)

X = mu.resblock_vqvae(X, 3, filter_nums[0], activation)
X = mu.resblock_vqvae(X, 3, filter_nums[0], activation)

X = layers.Conv2D(filter_nums[1], 3, strides=2, padding="same")(X)
X = layers.BatchNormalization()(X)
X = layers.Activation(activation)(X)

X = mu.resblock_vqvae(X, 3, filter_nums[1], activation)
X = mu.resblock_vqvae(X, 3, filter_nums[1], activation)

encoder_out = layers.Conv2D(latent_dim, 1, padding="same")(X)

# # --- VQ layer config --- #
vq_layer = vu.VectorQuantizer(num_embeddings, latent_dim)
X_VQ = vq_layer(encoder_out)
# # --- VQ layer config --- #

model_encoder = keras.Model(encoder_in, X_VQ)

# ---------------- decoder ----------------- #

decoder_in = keras.Input(shape=latent_size)

X = decoder_in

X = layers.Conv2D(filter_nums[1], 1, padding="same")(X)
X = layers.BatchNormalization()(X)
X = layers.Activation(activation)(X)

X = layers.Conv2DTranspose(filter_nums[1], 3, strides=2, padding="same")(X)
X = layers.BatchNormalization()(X)
X = layers.Activation(activation)(X)

X = mu.resblock_vqvae(X, 3, filter_nums[1], activation)
X = mu.resblock_vqvae(X, 3, filter_nums[1], activation)

X = layers.Conv2DTranspose(filter_nums[0], 3, strides=2, padding="same")(X)
X = layers.BatchNormalization()(X)
X = layers.Activation(activation)(X)

X = mu.resblock_vqvae(X, 3, filter_nums[0], activation)
X = mu.resblock_vqvae(X, 3, filter_nums[0], activation)

decoder_out = layers.Conv2D(1, 1, padding="same")(X)

model_decoder = keras.Model(decoder_in, decoder_out)

# ---------------- VQ-VAE ------------------ #
IN = keras.Input(shape=input_size)
X = IN
X_VQ = model_encoder(X)
# # --- VQ layer config --- #
# vq_layer = vu.VectorQuantizer(num_embeddings, latent_dim)
# X_VQ = vq_layer(X_encode)
# # --- VQ layer config --- #
OUT = model_decoder(X_VQ)
model_vqvae = keras.Model(IN, OUT)

# subclass to VAE training
vqvae_trainer = vu.VQVAETrainer(model_vqvae, 1.0, latent_dim, num_embeddings)

# load weights
if load_weights:
    W_old = mu.dummy_loader(model_name_load)
    vqvae_trainer.vqvae.set_weights(W_old)

# compile
vqvae_trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

# =================== Validation set preparation ===================== #

# location of training data
BATCH_dir = '/glade/campaign/cisl/aiml/ksha/BATCH_MRMS_025/'
# validation set size
L_valid = 500

# collect validation set sampales
filenames = np.array(sorted(glob(BATCH_dir+'*2023*.npy')))
L = len(filenames)
filename_valid = filenames[::14][:L_valid]

# MRMS batches were not normalized
Y_valid = np.empty((L_valid, 128, 256, 1))
Y_valid[...] = np.nan

for i, name in enumerate(filename_valid):
    Y_valid[i, ..., 0] = precip_norm(np.load(name))


# ======================== Model training ======================== #

filename_train1 = sorted(glob(BATCH_dir+'*2021*.npy'))
filename_train2 = sorted(glob(BATCH_dir+'*2022*.npy'))
filename_train = filename_train1 + filename_train2
L_train = len(filename_train)

min_del = 0.0
max_tol = 3 # early stopping with 2-epoch patience
tol = 0

Y_batch = np.empty((batch_size, 128, 256, 1))
Y_batch[...] = np.nan

for i in range(epochs):
    
    # # model training with on-the-fly batch generation
    # BATCH_dir = '/glade/campaign/cisl/aiml/ksha/BATCH_GFS_MRMS/'
    # filename_train = sorted(glob(BATCH_dir+'*.npy'))
    # L_train = len(filename_train)
    
    print('epoch = {}'.format(i))
    if i == 0:
        model_ = vqvae_trainer.vqvae
        Y_pred = model_.predict(Y_valid)
        Y_pred[Y_pred<0] = 0
        record = du.mean_absolute_error(Y_valid, Y_pred)
        print('Initial validation loss: {}'.format(record))
    
    start_time = time.time()
    for j in range(N_batch):
        
        inds_rnd = du.shuffle_ind(L_train)
        inds_ = inds_rnd[:batch_size]

        for k, ind in enumerate(inds_):
            # import batch data
            name = filename_train[ind]
            Y_batch[k, ..., 0] = precip_norm(np.load(name))
            
        vqvae_trainer.fit(Y_batch, epochs=2, batch_size=16, verbose=0)
        
    # on epoch-end
    model_ = vqvae_trainer.vqvae
    Y_pred = model_.predict(Y_valid)
    Y_pred[Y_pred<0] = 0
    record_temp = du.mean_absolute_error(Y_valid, Y_pred)

    if record - record_temp > min_del:
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        model_ = vqvae_trainer.vqvae
        print("Save to {}".format(model_name_save))
        model_.save(model_name_save)
        
    else:
        print('Validation loss {} NOT improved'.format(record_temp))

    print("--- %s seconds ---" % (time.time() - start_time))
    # mannual callbacks







