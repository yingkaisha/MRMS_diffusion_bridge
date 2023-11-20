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

epochs = 999999
filter_nums = [64, 128]
latent_dim = 64
num_embeddings = 128
activation = 'gelu'
model_name_save = '/glade/work/ksha/GAN/models/VQ_VAE_{}_{}_L{}_N{}_{}'.format(filter_nums[0], 
                                                                               filter_nums[1], 
                                                                               latent_dim, 
                                                                               num_embeddings, 
                                                                               activation)

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

class VQVAETrainer(keras.models.Model):
    def __init__(self, model, train_variance, latent_dim=32, num_embeddings=128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = model

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }



input_size = (256, 576, 1) #(28, 28, 1) #
latent_size = (64, 144, latent_dim) #(7, 7, latent_dim) #
# 64 144

# encoder
encoder_in = keras.Input(shape=input_size)
X = encoder_in

X = layers.Conv2D(filter_nums[0], 3, strides=2, padding="same")(X)
X = layers.BatchNormalization()(X)
X = layers.Activation(activation)(X)

X = layers.Conv2D(filter_nums[1], 3, strides=2, padding="same")(X)
X = layers.BatchNormalization()(X)
X = layers.Activation(activation)(X)

encoder_out = layers.Conv2D(latent_dim, 1, padding="same")(X)
model_encoder = keras.Model(encoder_in, encoder_out)

# decoder
decoder_in = keras.Input(shape=latent_size)
X = decoder_in
X = layers.Conv2DTranspose(filter_nums[1], 3, strides=2, padding="same")(X)
X = layers.BatchNormalization()(X)
X = layers.Activation(activation)(X)

X = layers.Conv2DTranspose(filter_nums[0], 3, strides=2, padding="same")(X)
X = layers.BatchNormalization()(X)
X = layers.Activation(activation)(X)

decoder_out = layers.Conv2DTranspose(1, 3, padding="same")(X)
model_decoder = keras.Model(decoder_in, decoder_out)

# VQ-VAE
IN = keras.Input(shape=input_size)
X = IN
X_encode = model_encoder(X)

vq_layer = VectorQuantizer(num_embeddings, latent_dim)
X_VQ = vq_layer(X_encode)
OUT = model_decoder(X_VQ)
model_vqvae = keras.Model(IN, OUT)

vqvae_trainer = VQVAETrainer(model_vqvae, 1.0, latent_dim, num_embeddings)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

def precip_norm(x):
    return np.log(x+1)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

BATCH_dir = '/glade/campaign/cisl/aiml/ksha/BATCH_MRMS/'

L_valid = 80

filenames = np.array(sorted(glob(BATCH_dir+'*.npy')))
L = len(filenames)

filename_valid = filenames[::10][:L_valid]
filename_train = list(set(filenames) - set(filename_valid))
L_train = len(filename_train)

Y_valid = np.empty((L_valid, 256, 576, 1))
for i, name in enumerate(filename_valid):
    Y_valid[i, ..., 0] = precip_norm(np.load(name))


L_train = 64
batch_size = 32

min_del = 0.0
max_tol = 3 # early stopping with 2-epoch patience
tol = 0

Y_batch = np.empty((batch_size, 256, 576, 1))
Y_batch[...] = np.nan

for i in range(epochs):
    print('epoch = {}'.format(i))
    if i == 0:
        model_ = vqvae_trainer.vqvae
        Y_pred = model_.predict(Y_valid)
        record = mean_absolute_error(Y_valid, Y_pred)
        print('Initial validation loss: {}'.format(record))
    
    start_time = time.time()
    for j in range(L_train):
        inds_rnd = du.shuffle_ind(L_train)
        inds_ = inds_rnd[:batch_size]

        for k, ind in enumerate(inds_):
            # import batch data
            temp_name = filename_train[ind]
            Y_batch[k, ..., 0] = precip_norm(np.load(temp_name))
            
        vqvae_trainer.fit(Y_batch, epochs=2, batch_size=16, verbose=0)
        
    # on epoch-end
    model_ = vqvae_trainer.vqvae
    Y_pred = model_.predict(Y_valid)
    record_temp = mean_absolute_error(Y_valid, Y_pred)

    if record - record_temp > min_del:
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        model_ = vqvae_trainer.vqvae
        model_.save(model_name_save)
        
    else:
        print('Validation loss {} NOT improved'.format(record_temp))

    print("--- %s seconds ---" % (time.time() - start_time))
    # mannual callbacks

