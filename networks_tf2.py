import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class C_VAE_NET(tf.keras.Model):
    def __init__(self, batch_size, time_steps, dim, z_dim,
                 enc_size, dec_size, enc_layers, dec_layers,
                 keep_prob, l2scale, conditional=False, num_labels=0):
        super(C_VAE_NET, self).__init__()
        
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.keep_prob = keep_prob
        self.l2scale = l2scale
        self.conditional = conditional
        self.num_labels = num_labels

        # Encoder layers
        self.encoder_layers = []
        for i in range(self.enc_layers):
            self.encoder_layers.append(
                tf.keras.layers.LSTM(
                    units=self.enc_size,
                    return_sequences=True,
                    dropout=1-self.keep_prob,
                    name=f"Continuous_VAE_{i}"
                )
            )
        
        # Decoder layers
        self.decoder_layers = []
        for i in range(self.dec_layers):
            self.decoder_layers.append(
                tf.keras.layers.LSTM(
                    units=self.dec_size,
                    return_sequences=True,
                    dropout=1-self.keep_prob,
                    name=f"Continuous_VAE_Dec_{i}"
                )
            )
        
        # Latent space projections
        self.z_mean_layer = tf.keras.layers.Dense(self.z_dim)
        self.z_log_var_layer = tf.keras.layers.Dense(self.z_dim)
        
        # Output projection
        self.output_layer = tf.keras.layers.Dense(self.dim)
        
        # Layer regularizers
        self.regularizer = tf.keras.regularizers.l2(l2scale)
    
    def encode(self, x, training=True):
        h = x
        for encoder_layer in self.encoder_layers:
            h = encoder_layer(h, training=training)
        return h
    
    def decode(self, z, training=True):
        h = z
        for decoder_layer in self.decoder_layers:
            h = decoder_layer(h, training=training)
        return self.output_layer(h)
    
    def reparameterize(self, mean, log_var):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[2]
        epsilon = tf.random.normal(shape=(batch, self.time_steps, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon
    
    def call(self, inputs, training=True):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            x, conditions = inputs
        else:
            x = inputs
            conditions = None
        
        # Encoding
        h_encoded = self.encode(x, training=training)
        
        # Latent representations
        z_mean = self.z_mean_layer(h_encoded)
        z_log_var = self.z_log_var_layer(h_encoded)
        
        # Sample using reparameterization trick
        z = self.reparameterize(z_mean, z_log_var)
        
        # Add conditions if needed
        if self.conditional and conditions is not None:
            # Replicate conditions across time dimension
            conditions = tf.tile(
                tf.expand_dims(conditions, axis=1),
                [1, self.time_steps, 1]
            )
            z = tf.concat([z, conditions], axis=-1)
        
        # Decoding
        reconstructed = self.decode(z, training=training)
        
        return reconstructed, tf.exp(z_log_var), z_mean, z_log_var, z


class D_VAE_NET(tf.keras.Model):
    def __init__(self, batch_size, time_steps, dim, z_dim,
                 enc_size, dec_size, enc_layers, dec_layers,
                 keep_prob, l2scale, conditional=False, num_labels=0):
        super(D_VAE_NET, self).__init__()
        
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.keep_prob = keep_prob
        self.l2scale = l2scale
        self.conditional = conditional
        self.num_labels = num_labels

        # Encoder layers
        self.encoder_layers = []
        for i in range(self.enc_layers):
            self.encoder_layers.append(
                tf.keras.layers.LSTM(
                    units=self.enc_size,
                    return_sequences=True,
                    dropout=1-self.keep_prob,
                    name=f"Discrete_VAE_{i}"
                )
            )
        
        # Decoder layers
        self.decoder_layers = []
        for i in range(self.dec_layers):
            self.decoder_layers.append(
                tf.keras.layers.LSTM(
                    units=self.dec_size,
                    return_sequences=True,
                    dropout=1-self.keep_prob,
                    name=f"Discrete_VAE_Dec_{i}"
                )
            )
        
        # Latent space projections
        self.z_mean_layer = tf.keras.layers.Dense(self.z_dim)
        self.z_log_var_layer = tf.keras.layers.Dense(self.z_dim)
        
        # Output projection with sigmoid for discrete data
        self.output_layer = tf.keras.layers.Dense(self.dim, activation='sigmoid')
        
        # Layer regularizers
        self.regularizer = tf.keras.regularizers.l2(l2scale)
    
    def encode(self, x, training=True):
        h = x
        for encoder_layer in self.encoder_layers:
            h = encoder_layer(h, training=training)
        return h
    
    def decode(self, z, training=True):
        h = z
        for decoder_layer in self.decoder_layers:
            h = decoder_layer(h, training=training)
        return self.output_layer(h)
    
    def reparameterize(self, mean, log_var):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[2]
        epsilon = tf.random.normal(shape=(batch, self.time_steps, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon
    
    def call(self, inputs, training=True):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            x, conditions = inputs
        else:
            x = inputs
            conditions = None
        
        # Encoding
        h_encoded = self.encode(x, training=training)
        
        # Latent representations
        z_mean = self.z_mean_layer(h_encoded)
        z_log_var = self.z_log_var_layer(h_encoded)
        
        # Sample using reparameterization trick
        z = self.reparameterize(z_mean, z_log_var)
        
        # Add conditions if needed
        if self.conditional and conditions is not None:
            # Replicate conditions across time dimension
            conditions = tf.tile(
                tf.expand_dims(conditions, axis=1),
                [1, self.time_steps, 1]
            )
            z = tf.concat([z, conditions], axis=-1)
        
        # Decoding
        reconstructed = self.decode(z, training=training)
        
        return reconstructed, tf.exp(z_log_var), z_mean, z_log_var, z


class Generator(tf.keras.Model):
    def __init__(self, time_steps, noise_dim, output_dim, num_units, num_layers, 
                 keep_prob, conditional=False, num_labels=0):
        super(Generator, self).__init__()
        
        self.time_steps = time_steps
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.num_units = num_units
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.conditional = conditional
        self.num_labels = num_labels
        
        # LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    units=num_units,
                    return_sequences=True,
                    dropout=1-keep_prob,
                    name=f"Generator_LSTM_{i}"
                )
            )
            
        # Output projection
        self.output_layer = tf.keras.layers.Dense(output_dim)
        
    def call(self, inputs, training=True):
        if self.conditional and isinstance(inputs, tuple):
            noise, conditions = inputs
            # Replicate conditions across time dimension
            conditions = tf.tile(
                tf.expand_dims(conditions, axis=1),
                [1, self.time_steps, 1]
            )
            # Concatenate noise and conditions
            h = tf.concat([noise, conditions], axis=-1)
        else:
            h = inputs
            
        # Pass through LSTM layers
        for lstm_layer in self.lstm_layers:
            h = lstm_layer(h, training=training)
            
        # Project to output dimension
        return self.output_layer(h)


class Discriminator(tf.keras.Model):
    def __init__(self, num_units, num_layers, keep_prob):
        super(Discriminator, self).__init__()
        
        self.num_units = num_units
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        
        # LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    units=num_units,
                    return_sequences=True,
                    dropout=1-keep_prob,
                    name=f"Discriminator_LSTM_{i}"
                )
            )
            
        # Output projection for classification (real/fake)
        self.output_layer = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=True):
        h = inputs
        
        # Pass through LSTM layers
        for lstm_layer in self.lstm_layers:
            h = lstm_layer(h, training=training)
            
        # Project to output dimension
        return self.output_layer(h)


class C_GAN_NET:
    def __init__(self, batch_size, noise_dim, dim, gen_dim, time_steps,
                 gen_num_units, gen_num_layers, dis_num_units, dis_num_layers,
                 keep_prob, l2_scale, conditional=False, num_labels=0):
        
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.dim = dim
        self.gen_dim = gen_dim
        self.time_steps = time_steps
        self.gen_num_units = gen_num_units
        self.gen_num_layers = gen_num_layers
        self.dis_num_units = dis_num_units
        self.dis_num_layers = dis_num_layers
        self.keep_prob = keep_prob
        self.l2_scale = l2_scale
        self.conditional = conditional
        self.num_labels = num_labels
        
        # Create generator and discriminator
        self.generator = Generator(
            time_steps=time_steps,
            noise_dim=noise_dim,
            output_dim=dim,
            num_units=gen_num_units,
            num_layers=gen_num_layers,
            keep_prob=keep_prob,
            conditional=conditional,
            num_labels=num_labels
        )
        
        self.discriminator = Discriminator(
            num_units=dis_num_units,
            num_layers=dis_num_layers,
            keep_prob=keep_prob
        )


class D_GAN_NET:
    def __init__(self, batch_size, noise_dim, dim, gen_dim, time_steps,
                 gen_num_units, gen_num_layers, dis_num_units, dis_num_layers,
                 keep_prob, l2_scale, conditional=False, num_labels=0):
        
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.dim = dim
        self.gen_dim = gen_dim
        self.time_steps = time_steps
        self.gen_num_units = gen_num_units
        self.gen_num_layers = gen_num_layers
        self.dis_num_units = dis_num_units
        self.dis_num_layers = dis_num_layers
        self.keep_prob = keep_prob
        self.l2_scale = l2_scale
        self.conditional = conditional
        self.num_labels = num_labels
        
        # Create generator and discriminator with sigmoid activation for discrete data
        self.generator = Generator(
            time_steps=time_steps,
            noise_dim=noise_dim,
            output_dim=dim,
            num_units=gen_num_units,
            num_layers=gen_num_layers,
            keep_prob=keep_prob,
            conditional=conditional,
            num_labels=num_labels
        )
        
        self.discriminator = Discriminator(
            num_units=dis_num_units,
            num_layers=dis_num_layers,
            keep_prob=keep_prob
        )