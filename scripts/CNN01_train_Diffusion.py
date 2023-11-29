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

# ============================= Hyperparameters ============================= #

total_timesteps = 50 # diffusion time steps
norm_groups = 8 # number of attention heads, number of layer normalization groups 

# min-max values of the diffusion target (learning target) 
clip_min = -1.0
clip_max = 1.0

Fy = 1/np.log(100+1) # max precip 100 mm

widths = [16, 32, 48, 64] # number of convolution kernels per up-/downsampling level
left_attention = [False, False, True, True] # True: use multi-head attnetion on each up-/downsampling level
right_attention = [False, False, True, True]
num_res_blocks = 1  # Number of residual blocks

input_shape = (128, 128, 2) # the tensor shape of reverse diffusion input

load_weights = True # True: load previous weights

# location of the previous weights
model_name = '/glade/work/ksha/GAN/models/DM_example_tune/'
# location for saving new weights
model_name_save = '/glade/work/ksha/GAN/models/DM_example_tune2/'

lr = 1e-5 # learning rate

# samples per epoch = N_batch * batch_size
epochs = 99999
N_batch = 128
batch_size = 32

# ============================= Model design ============================= #

def build_model(input_shape, widths, left_attention, right_attention, num_res_blocks=2, norm_groups=8,
                interpolation="nearest", activation_fn=keras.activations.swish,):

    first_conv_channels = widths[0]
    
    image_input = layers.Input(shape=input_shape, name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    
    x = layers.Conv2D(first_conv_channels, kernel_size=(3, 3), padding="same",
                      kernel_initializer=mu.kernel_init(1.0),)(image_input)

    temb = mu.TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = mu.TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)

    skips = [x]

    # DownBlock
    has_attention = left_attention
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = mu.ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            
            if has_attention[i]:
                x = layers.MultiHeadAttention(num_heads=norm_groups, key_dim=widths[i])(x, x)
                
            skips.append(x)

        if widths[i] != widths[-1]:
            x = mu.DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = mu.ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = layers.MultiHeadAttention(num_heads=norm_groups, key_dim=widths[-1])(x, x)
    x = mu.ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])

    # UpBlock
    has_attention = right_attention
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = mu.ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            
            if has_attention[i]:
                x = layers.MultiHeadAttention(num_heads=norm_groups, key_dim=widths[i])(x, x)

        if i != 0:
            x = mu.UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=mu.kernel_init(0.0))(x)
    return keras.Model([image_input, time_input], x, name="unet")
    
# Reverse diffusino model
model = build_model(input_shape=input_shape, widths=widths,
                    left_attention=left_attention, right_attention=right_attention, 
                    num_res_blocks=num_res_blocks, norm_groups=norm_groups, activation_fn=keras.activations.swish)

# Compile the mdoel
model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(learning_rate=lr),)

# load previous weights
if load_weights:
    W_old = mu.dummy_loader(model_name)
    model.set_weights(W_old)

# configure the forward diffusion steps
gdf_util = mu.GaussianDiffusion(timesteps=total_timesteps)

# ======================================== Validation set preparation ===================================== #

# location of training data
BATCH_dir = '/glade/campaign/cisl/aiml/ksha/BATCH_GFS_VALID/'

# validation set size
L_valid = 500

# collect validation set sampales
filenames = np.array(sorted(glob(BATCH_dir+'*.npy')))
filename_valid = filenames[::7][:L_valid]

Y_valid = np.empty((L_valid,)+input_shape)

for i, name in enumerate(filename_valid):
    temp_data = np.load(name)
    Y_valid[i, ...] = Fy*temp_data[0, ..., :2]

# validate on random timesteps
t_valid_ = np.random.uniform(low=0, high=total_timesteps, size=(L_valid,))
t_valid = t_valid_.astype(int)

# sample random noise to be added to the images in the batch
noise_valid = np.random.normal(size=((L_valid,)+(128, 128, 1)))
images_valid_ = np.array(gdf_util.q_sample(Y_valid[..., 0][..., None], t_valid, noise_valid))
images_valid = np.concatenate((images_valid_, Y_valid[..., 1][..., None]), axis=-1)
# # validation prediction example:
# # pred_noise = model.predict([images_valid, t_valid, X_valid])

# ================================== Training loop ================================== #

# collect training samples
BATCH_dir = '/glade/campaign/cisl/aiml/ksha/BATCH_GFS_MRMS/'
filename_train = np.array(sorted(glob(BATCH_dir+'*.npy')))
L_train = len(filename_train)

min_del = 0.0
max_tol = 3 # early stopping with 2-epoch patience
tol = 0

Y_batch = np.empty((batch_size,)+input_shape)

for i in range(epochs):
    
    print('epoch = {}'.format(i))
    if i == 0:
        pred_noise = model.predict([images_valid, t_valid])
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
            temp_data = np.load(temp_name)
            Y_batch[k, ...] = Fy*temp_data[0, ..., :2]
        
        # sample timesteps uniformly
        t_ = np.random.uniform(low=0, high=total_timesteps, size=(batch_size,))
        t = t_.astype(int)
        
        # sample random noise to be added to the images in the batch
        noise = np.random.normal(size=(batch_size,)+input_shape)
        images_t = np.array(gdf_util.q_sample(Y_batch, t, noise))
        
        # sample random noise to be added to the images in the batch
        noise = np.random.normal(size=((batch_size,)+(128, 128, 1)))
        images_t_ = np.array(gdf_util.q_sample(Y_batch[..., 0][..., None], t, noise))
        images_t = np.concatenate((images_t_, Y_batch[..., 1][..., None]), axis=-1)
        
        # train on batch
        model.train_on_batch([images_t, t], noise)
        
    # on epoch-end
    pred_noise = model.predict([images_valid, t_valid])
    record_temp = np.mean(np.abs(noise_valid - pred_noise))
    
    # print out valid loss change
    if record - record_temp > min_del:
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        print("Save to {}".format(model_name_save))
        model.save(model_name_save)
        
    else:
        print('Validation loss {} NOT improved'.format(record_temp))

    print("--- %s seconds ---" % (time.time() - start_time))
    # mannual callbacks






