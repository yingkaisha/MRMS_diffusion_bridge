import os
import sys

import time
import numpy as np
from glob import glob
from scipy.stats import truncnorm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

train_round = 0
weights_round = -1

model_de_save = 'MyGAN_de{}'.format(train_round)
model_gen_save = 'MyGAN_gen{}'.format(train_round)

model_de_load = 'MyGAN_de{}'.format(weights_round)
model_gen_load = 'MyGAN_gen{}'.format(weights_round)

def mae(x1, x2):
    return np.mean(np.abs(x1-x2))

def pixel_norm(x, epsilon=1e-8):
    return x / tf.math.sqrt(tf.reduce_mean(x ** 2, axis=-1, keepdims=True) + epsilon)

class AddNoise(layers.Layer):
    def build(self, input_shape):
        n, h, w, c = input_shape[0]
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        self.b = self.add_weight(
            shape=[1, 1, 1, c], initializer=initializer, trainable=True, name="kernel"
        )

    def call(self, inputs):
        x, noise = inputs
        output = x + self.b * noise
        return output


class AdaIN(layers.Layer):
    def __init__(self, gain=1, **kwargs):
        super().__init__(**kwargs)
        self.gain = gain

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]

        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]

        self.dense_1 = keras.layers.Dense(self.x_channels)
        self.dense_2 = keras.layers.Dense(self.x_channels)

    def call(self, inputs):
        x, w = inputs
        ys = tf.reshape(self.dense_1(w), (-1, 1, 1, self.x_channels))
        yb = tf.reshape(self.dense_2(w), (-1, 1, 1, self.x_channels))
        return ys * x + yb

size_input_vector = 120
size_latent_vector = 120

size_input_res = 8
size_input_channel = 120 #1
size_latent_channel = 120

size_output_res = 256
size_output_channel = 1

size_progress = [8, 16, 32, 64, 128, 256]
filter_nums = [size_latent_vector, size_latent_vector, size_latent_vector, size_latent_vector, size_latent_vector, size_latent_vector]

n_mapping = 8

def get_truncated_normal(mean=0, sd=1, low=-2.5, upp=2.5):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

norm_gen = get_truncated_normal()

filenames = np.array(sorted(glob(BATCH_dir+'DSCALE*.npy')))

L = len(filenames)
filename_valid = filenames[::20][:1000]
filename_train = list(set(filenames) - set(filename_valid))
L_valid = len(filename_valid)
L_train = len(filename_train)

X_valid_profile = np.zeros((L_valid, size_input_vector))
X_valid_LR = np.zeros((L_valid, size_input_res, size_input_res, size_input_channel))
Y_valid_HR = np.empty((L_valid, size_output_res, size_output_res, size_output_channel))

for i, name in enumerate(filename_valid):
    temp_data = np.load(name, allow_pickle=True)[()]

    Y_valid_HR[i, ...] = temp_data['patch'][..., None]

    X_valid_profile[i, 0:12] = temp_data['profile_t']
    X_valid_profile[i, 12:24] = temp_data['profile_tmax']
    X_valid_profile[i, 24:36] = temp_data['profile_u']
    X_valid_profile[i, 36:48] = temp_data['profile_umax']
    X_valid_profile[i, 48:60] = temp_data['profile_v']
    X_valid_profile[i, 60:72] = temp_data['profile_vmax']
    X_valid_profile[i, 72:84] = temp_data['profile_rh']
    X_valid_profile[i, 84:96] = temp_data['profile_rhmax']
    X_valid_profile[i, 96:108] = temp_data['profile_gph']
    X_valid_profile[i, 108:120] = temp_data['profile_gphmax']

    X_valid_LR[i, :, :, :] = temp_data['context'][..., None]

input_noise = []
for res in size_progress:
    input_noise.append(norm_gen.rvs(size=(L_valid, res, res, 1)))

def Mapping(num_stages, latent_shape):
    '''
    The mapping network
    Input noise --> Latent space projection
    '''
    z = layers.Input(shape=(latent_shape))
    w = pixel_norm(z)
    
    for i in range(8):
        w = keras.layers.Dense(latent_shape)(w)
        w = keras.activations.gelu(w)
        
    w = tf.tile(tf.expand_dims(w, 1), (1, num_stages, 1))
    
    return keras.Model(z, w, name="mapping")

def generator(size_input_res, size_input_channel, size_latent_channel, filter_nums, size_progress):
    '''
    Generator backbone
    '''
    g_input = layers.Input((size_input_res, size_input_res, size_input_channel))
    x = g_input
    
    w = layers.Input(shape=(len(filter_nums), size_latent_channel))
    
    noise_branch = []
    
    for i in range(len(filter_nums)):
        
        filter_num = filter_nums[i]
        res = size_progress[i]
        
        # noise inputs from the right side
        noise_input = layers.Input(shape=(res, res, 1))
        noise_branch.append(noise_input)

        # upsampling starts from the second block
        if i > 0:
            x = layers.UpSampling2D((2, 2))(x)

        for _ in range(2):
            # Conv --> add noise --> activation --> Instance norm --> AdaIN
            x = keras.layers.Conv2D(filter_num, 3, padding="same")(x)
            x = AddNoise()([x, noise_branch[i]])
            x = keras.activations.gelu(x)
            x = layers.BatchNormalization(axis=[1, 2])(x, training=True)
            x = AdaIN()([x, w[:, i]])
            
    out = x
    out = keras.layers.Conv2D(1, 1, padding="valid")(out)
    
    generator_backbone = keras.Model([g_input, w,]+noise_branch, out)
    return generator_backbone

def discriminator(size_output_res, size_output_channel):
    d_input = layers.Input((size_output_res, size_output_res, size_output_channel))
    x = d_input
    
    x = keras.layers.Conv2D(16, 1, padding="same")(x)
    x = keras.activations.gelu(x)
    x = keras.layers.Conv2D(16, 3, padding="same")(x)
    x = keras.activations.gelu(x)
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.activations.gelu(x)
    
    x = layers.AveragePooling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, 3, padding="same")(x)
    x = keras.activations.gelu(x)
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.activations.gelu(x)
    
    x = layers.AveragePooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.activations.gelu(x)
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.activations.gelu(x)
    
    x = layers.AveragePooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.activations.gelu(x)
    x = keras.layers.Conv2D(256, 3, padding="same")(x)
    x = keras.activations.gelu(x)
    
    x = layers.AveragePooling2D((2, 2))(x)
    x = keras.layers.Conv2D(256, 3, padding="same")(x)
    x = keras.activations.gelu(x)
    x = keras.layers.Conv2D(512, 3, padding="same")(x)
    x = keras.activations.gelu(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    d_out = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(d_input, d_out)

# mapping network
mapping_network = Mapping(len(filter_nums), size_latent_vector)

# generator backbone
generator_backbone = generator(size_input_res, size_input_channel, size_latent_channel, filter_nums, size_progress)

IN_vector = layers.Input(shape=(size_latent_vector))
w_map = mapping_network(IN_vector)

noise_branch = []
for i in range(len(filter_nums)):
    res = size_progress[i]
    noise_input = layers.Input(shape=(res, res, 1))
    noise_branch.append(noise_input)
    
IN_LR = layers.Input((size_input_res, size_input_res, size_input_channel))

g_out = generator_backbone([IN_LR, w_map,]+noise_branch)

# merge the two above
StyleGAN_generator = keras.Model([IN_LR, IN_vector,]+noise_branch, g_out)
StyleGAN_discriminator = discriminator(size_output_res, size_output_channel)

IN_LR = layers.Input((size_input_res, size_input_res, size_input_channel))
IN_vector = layers.Input(shape=(size_latent_vector))

noise_branch = []
for i in range(len(filter_nums)):
    res = size_progress[i]
    noise_input = layers.Input(shape=(res, res, 1))
    noise_branch.append(noise_input)

IN_package = [IN_LR, IN_vector,]+noise_branch

G_OUT = StyleGAN_generator(IN_package)
D_OUT = StyleGAN_discriminator(G_OUT)

StyleGAN = keras.models.Model(IN_package, [G_OUT, D_OUT])

def wasserstein_loss(y_true, y_pred):
    return 1-tf.reduce_mean(y_true * y_pred)

def dssim_mae(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=5))
    mae_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return ssim_loss+mae_loss

StyleGAN_discriminator.compile(loss=wasserstein_loss, optimizer=keras.optimizers.Adam(lr=1e-4, beta_1=0, beta_2=0.999))
StyleGAN_generator.compile(loss=dssim_mae, optimizer=keras.optimizers.Adam(lr=1e-4, beta_1=0, beta_2=0.999))

StyleGAN.compile(loss=[dssim_mae, wasserstein_loss], loss_weights=[1.0, 1.0], 
                 optimizer=keras.optimizers.Adam(lr=1e-4, beta_1=0, beta_2=0.999))

if weights_round >= 0:
    W_old = mu.dummy_loader('/glade/work/ksha/GAN/models/{}/'.format(model_de_load))
    StyleGAN_discriminator.set_weights(W_old)
    W_old = mu.dummy_loader('/glade/work/ksha/GAN/models/{}/'.format(model_gen_load))
    StyleGAN_generator.set_weights(W_old)

VALID = StyleGAN_generator.predict([X_valid_LR, X_valid_profile]+input_noise)
record = mae(VALID, Y_valid_HR)

epochs = 99999
L_train = 512
batch_size = 32

min_del = 0.0
max_tol = 3 # early stopping with 2-epoch patience
tol = 0

# loss backup
GAN_LOSS = np.zeros([int(epochs*L_train), 3])*np.nan
D_LOSS = np.zeros([int(epochs*L_train)])*np.nan
V_LOSS = np.zeros([epochs])

X_batch_profile = np.empty((batch_size, size_input_vector))
X_batch_profile[...] = np.nan
X_batch_LR = np.empty((batch_size, size_input_res, size_input_res, size_input_channel))
X_batch_LR[...] = np.nan
Y_batch_HR = np.empty((batch_size, size_output_res, size_output_res, size_output_channel))
Y_batch_HR[...] = np.nan

y_bad = np.zeros(batch_size)
y_good = np.random.uniform(0.95, 0.999, size=batch_size) #np.ones(batch_size)
dummy_good = y_good
dummy_mix = np.concatenate((y_bad, y_good), axis=0)

for i in range(epochs):
    print('epoch = {}'.format(i))
    if i == 0:
        print('Initial validation loss: {}'.format(record))
        
    start_time = time.time()
    # loop over batches
    
    for j in range(L_train):
        
        inds_rnd = du.shuffle_ind(L_train)
        inds_ = inds_rnd[:batch_size]
        
        for k, ind in enumerate(inds_):
            # import batch data
            temp_name = filename_train[ind]
            temp_data = np.load(temp_name, allow_pickle=True)[()]
    
            Y_batch_HR[k, ...] = temp_data['patch'][..., None]
            
            X_batch_profile[k, 0:12] = temp_data['profile_t']
            X_batch_profile[k, 12:24] = temp_data['profile_tmax']
            X_batch_profile[k, 24:36] = temp_data['profile_u']
            X_batch_profile[k, 36:48] = temp_data['profile_umax']
            X_batch_profile[k, 48:60] = temp_data['profile_v']
            X_batch_profile[k, 60:72] = temp_data['profile_vmax']
            X_batch_profile[k, 72:84] = temp_data['profile_rh']
            X_batch_profile[k, 84:96] = temp_data['profile_rhmax']
            X_batch_profile[k, 96:108] = temp_data['profile_gph']
            X_batch_profile[k, 108:120] = temp_data['profile_gphmax']
            
            X_batch_LR[k, :, :, :] = temp_data['context'][..., None]
        
        batch_noise = []
        for res in size_progress:
            batch_noise.append(norm_gen.rvs(size=(batch_size, res, res, 1)))

        #
        N = np.sum(np.isnan(Y_batch_HR)) + np.sum(np.isnan(X_batch_profile)) + np.sum(np.isnan(X_batch_LR))
        if N > 0:
            print('aergeqagr')
            continue;
        #
        
        # get G_output
        StyleGAN_discriminator.trainable = True
        g_out = StyleGAN_generator.predict([X_batch_LR, X_batch_profile]+batch_noise) # <-- np.array

        # test D with G_output
        d_in_Y = np.concatenate((g_out, Y_batch_HR), axis=0) # batch size doubled
        #d_in_X = np.concatenate((X_batch, X_batch), axis=0)
        d_target = dummy_mix
        
        batch_ind = du.shuffle_ind(2*batch_size)
        d_loss = StyleGAN_discriminator.train_on_batch(d_in_Y[batch_ind, ...], d_target[batch_ind, ...])

        # G training / transferring
        StyleGAN_discriminator.trainable = False
        gan_in = [X_batch_LR, X_batch_profile]+batch_noise
        gan_target = [Y_batch_HR, dummy_good]
        #gan_target = [dummy_good,]

        gan_loss = StyleGAN.train_on_batch(gan_in, gan_target)
        # # Backup training loss
        D_LOSS[i*L_train+j] = d_loss
        GAN_LOSS[i*L_train+j, :] = gan_loss
        #
        if j%10 == 0:
            print('\t{} step loss = {}'.format(j, gan_loss))
    # on epoch-end
    
    # save model regardless
    StyleGAN_discriminator.save('/glade/work/ksha/GAN/models/{}/'.format(model_de_save))
    StyleGAN_generator.save('/glade/work/ksha/GAN/models/{}/'.format(model_gen_save))

    VALID = StyleGAN_generator.predict([X_valid_LR, X_valid_profile]+input_noise)
    record_temp = mae(VALID, Y_valid_HR)
    # Backup validation loss
    V_LOSS[i] = record_temp
    
    # print out valid loss change
    if record - record_temp > min_del:
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
    else:
        print('Validation loss {} NOT improved'.format(record_temp))
        
    save_dict = {}
    save_dict['D_LOSS'] = D_LOSS
    save_dict['GAN_LOSS'] = GAN_LOSS
    save_dict['V_LOSS'] = V_LOSS
    save_loss_name = '/glade/work/ksha/GAN/models/LOSS_{}.npy'.format(model_gen_save)
    np.save(save_loss_name, save_dict)
    
    print(save_loss_name)
    print("--- %s seconds ---" % (time.time() - start_time))
    # mannual callbacks



