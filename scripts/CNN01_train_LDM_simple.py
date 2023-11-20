import os
import sys
import time
import math
import logging
import warnings
import numpy as np
from glob import glob

# supress regular warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("tensorflow").setLevel(logging.ERROR) 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# supress tensorflow warnings
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

# adjust for time step embedding layer
tf.config.run_functions_eagerly(True)

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

# ================================================= #

total_timesteps = 100 # diffusion time steps
norm_groups = 8 # number of attention heads, number of layer normalization groups 
lr = 1e-4 # learning rate

# min-max values of the diffusion target (learning target) 
clip_min = -1.0
clip_max = 1.0

widths = [64, 96, 128, 256] # number of convolution kernels per up-/downsampling level
has_attention = [False, False, False, True] # True: use multi-head attnetion on each up-/downsampling level
num_res_blocks = 1  # Number of residual blocks

input_shape = (32, 32, 16) # the tensor shape of reverse diffusion input
gfs_shape = (32, 32, 256) # the tensor shape of GFS embeddings

F_x = 0.1 # the scale of GFS embeddings
F_y = 1/2.76 # the scale of VQ-VAE codes

load_weights = True # True: load previous weights

# location of the previous weights
model_name = '/glade/work/ksha/GAN/models/LDM_atten{}_res{}_base/'.format(4, 1)

# location for saving new weights
model_name_save = '/glade/work/ksha/GAN/models/LDM_atten{}_res{}_tune/'.format(1, 1)

# ================================================= #

# Reverse diffusino model
model = mu.build_model(input_shape=input_shape, gfs_shape=gfs_shape, widths=widths,
                       has_attention=has_attention, num_res_blocks=num_res_blocks, 
                       norm_groups=norm_groups, activation_fn=keras.activations.swish)

# Compile the mdoel
model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(learning_rate=lr),)

# load previous weights
if load_weights:
    W_old = mu.dummy_loader(model_name)
    model.set_weights(W_old)

# configure the forward diffusion steps
gdf_util = mu.GaussianDiffusion(timesteps=total_timesteps)

# =================== Validation set ====================== #
L_valid = 32 # number of validation samples

# locations of training data
BATCH_dir = '/glade/campaign/cisl/aiml/ksha/BATCH_LDM/'

# preparing training batches
filenames = np.array(sorted(glob(BATCH_dir+'*.npy')))

L = len(filenames)
filename_valid = filenames[:][:L_valid]
filename_train = list(set(filenames) - set(filename_valid))

L_train = len(filename_train)

Y_valid = np.empty((L_valid, 32, 32, 16))
X_valid = np.empty((L_valid, 32, 32, 256))

for i, name in enumerate(filename_valid):
    temp_data = np.load(name, allow_pickle=True)[()]
    X_valid[i, ...] = F_x*temp_data['GFS_latent']
    Y_valid[i, ...] = F_y*temp_data['Y_latent']

# validate on random timesteps
t_valid_ = np.random.uniform(low=0, high=total_timesteps, size=(L_valid,))
t_valid = t_valid_.astype(int)

# sample random noise to be added to the images in the batch
noise_valid = np.random.normal(size=(L_valid, 32, 32, 16))
images_valid = np.array(gdf_util.q_sample(Y_valid, t_valid, noise_valid))

# validation prediction example:
# pred_noise = model.predict([images_valid, t_valid, X_valid])

# =================== Training loop ====================== #

# collect all training batches
filenames = np.array(sorted(glob(BATCH_dir+'*.npy')))
filename_valid = filenames[:L_valid]
filename_train = list(set(filenames) - set(filename_valid))
L_train = len(filename_train)

# samples per epoch = N_batch * batch_size
epochs = 99999
N_batch = 128
batch_size = 32

min_del = 0.0
max_tol = 3 # early stopping with 2-epoch patience
tol = 0

Y_batch = np.empty((batch_size, 32, 32, 16))
X_batch = np.empty((batch_size, 32, 32, 256))

for i in range(epochs):
    
    print('epoch = {}'.format(i))
    if i == 0:
        pred_noise = model.predict([images_valid, t_valid, X_valid])
        record = np.mean(np.abs(noise_valid - pred_noise))
        #print('initial loss {}'.format(record))
        print('Initial validation loss: {}'.format(record))
        
    start_time = time.time()
    # loop over batches
    for j in range(N_batch):
        
        inds_rnd = du.shuffle_ind(L_train) # shuffle training files
        inds_ = inds_rnd[:batch_size] # select training files
        
        # collect training batches
        for k, ind in enumerate(inds_):
            # import batch data
            temp_name = filename_train[ind]
            temp_data = np.load(temp_name, allow_pickle=True)[()]
            X_batch[k, ...] = F_x*temp_data['GFS_latent']
            Y_batch[k, ...] = F_y*temp_data['Y_latent']

        # sample timesteps uniformly
        t_ = np.random.uniform(low=0, high=total_timesteps, size=(batch_size,))
        t = t_.astype(int)
        
        # sample random noise to be added to the images in the batch
        noise = np.random.normal(size=(batch_size, 32, 32, 16))
        images_t = np.array(gdf_util.q_sample(Y_batch, t, noise))
        
        # train on batch
        model.train_on_batch([images_t, t, X_batch], noise)
        
    # on epoch-end
    pred_noise = model.predict([images_valid, t_valid, X_valid])
    record_temp = np.mean(np.abs(noise_valid - pred_noise))
    
    # print out valid loss change
    if record - record_temp > min_del:
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        model.save(model_name_save)
        
    else:
        print('Validation loss {} NOT improved'.format(record_temp))

    print("--- %s seconds ---" % (time.time() - start_time))
    # mannual callbacks




