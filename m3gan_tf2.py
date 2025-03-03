import tensorflow as tf
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class m3gan:
    def __init__(self, batch_size, time_steps, num_pre_epochs, num_epochs,
                 checkpoint_dir, epoch_ckpt_freq, epoch_loss_freq,
                 # params for c
                 c_dim, c_noise_dim, c_z_size, c_data_sample, c_vae, c_gan,
                 # params for d
                 d_dim, d_noise_dim, d_z_size, d_data_sample, d_vae, d_gan,
                 # params for training
                 d_rounds=1, g_rounds=1, v_rounds=1,
                 v_lr_pre=0.001, v_lr=0.001, g_lr=0.001, d_lr=0.001,
                 alpha_re=1.0, alpha_kl=0.1, alpha_mt=2.0, alpha_ct=0.0,
                 alpha_sm=0.0, c_beta_adv=1.0, c_beta_fm=1.0,
                 d_beta_adv=1.0, d_beta_fm=1.0, conditional=False,
                 num_labels=0, statics_label=None):
        
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.num_pre_epochs = num_pre_epochs
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.epoch_ckpt_freq = epoch_ckpt_freq
        self.epoch_loss_freq = epoch_loss_freq
        
        # Params for continuous data
        self.c_dim = c_dim
        self.c_noise_dim = c_noise_dim
        self.c_z_size = c_z_size
        self.c_data_sample = c_data_sample
        self.c_vae = c_vae
        self.c_gan = c_gan
        
        # Params for discrete data
        self.d_dim = d_dim
        self.d_noise_dim = d_noise_dim
        self.d_z_size = d_z_size
        self.d_data_sample = d_data_sample
        self.d_vae = d_vae
        self.d_gan = d_gan
        
        # Params for training
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.v_rounds = v_rounds
        self.v_lr_pre = v_lr_pre
        self.v_lr = v_lr
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.alpha_re = alpha_re
        self.alpha_kl = alpha_kl
        self.alpha_mt = alpha_mt
        self.alpha_ct = alpha_ct
        self.alpha_sm = alpha_sm
        self.c_beta_adv = c_beta_adv
        self.c_beta_fm = c_beta_fm
        self.d_beta_adv = d_beta_adv
        self.d_beta_fm = d_beta_fm
        self.conditional = conditional
        self.num_labels = num_labels
        self.statics_label = statics_label
        
        # Create optimizers
        self.c_vae_optimizer = tf.keras.optimizers.Adam(learning_rate=v_lr)
        self.c_gen_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr)
        self.c_dis_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr)
        self.d_vae_optimizer = tf.keras.optimizers.Adam(learning_rate=v_lr)
        self.d_gen_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr)
        self.d_dis_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr)
        
        # Checkpointing
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            c_vae_optimizer=self.c_vae_optimizer,
            c_gen_optimizer=self.c_gen_optimizer,
            c_dis_optimizer=self.c_dis_optimizer,
            d_vae_optimizer=self.d_vae_optimizer,
            d_gen_optimizer=self.d_gen_optimizer,
            d_dis_optimizer=self.d_dis_optimizer,
            c_vae=self.c_vae,
            c_generator=self.c_gan.generator,
            c_discriminator=self.c_gan.discriminator,
            d_vae=self.d_vae,
            d_generator=self.d_gan.generator,
            d_discriminator=self.d_gan.discriminator
        )
        
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=5)
        
    def build(self):
        """Initialize models by passing some data through them"""
        print("Initializing models...")
        
        # Get sample shapes
        batch_size = self.batch_size
        time_steps = self.time_steps
        c_dim = self.c_dim
        d_dim = self.d_dim
        c_noise_dim = self.c_noise_dim
        d_noise_dim = self.d_noise_dim
        
        # Generate sample data to initialize model shapes
        sample_c_data = tf.random.normal([batch_size, time_steps, c_dim])
        sample_d_data = tf.random.uniform([batch_size, time_steps, d_dim])
        sample_c_noise = tf.random.normal([batch_size, time_steps, c_noise_dim])
        sample_d_noise = tf.random.normal([batch_size, time_steps, d_noise_dim])
        
        # Pass data through models to build them
        if self.conditional:
            sample_conditions = tf.random.normal([batch_size, self.num_labels])
            
            # Initialize VAEs
            self.c_vae([sample_c_data, sample_conditions], training=False)
            self.d_vae([sample_d_data, sample_conditions], training=False)
            
            # Initialize GANs
            self.c_gan.generator([sample_c_noise, sample_conditions], training=False)
            self.c_gan.discriminator(sample_c_data, training=False)
            
            self.d_gan.generator([sample_d_noise, sample_conditions], training=False)
            self.d_gan.discriminator(sample_d_data, training=False)
        else:
            # Initialize VAEs
            self.c_vae(sample_c_data, training=False)
            self.d_vae(sample_d_data, training=False)
            
            # Initialize GANs
            self.c_gan.generator(sample_c_noise, training=False)
            self.c_gan.discriminator(sample_c_data, training=False)
            
            self.d_gan.generator(sample_d_noise, training=False)
            self.d_gan.discriminator(sample_d_data, training=False)
        
        print("Models initialized successfully!")
    
    def generate_data(self, num_sample=1000):
        """Generate synthetic data using the trained models"""
        # Generate noise
        c_noise = tf.random.normal([num_sample, self.time_steps, self.c_noise_dim])
        d_noise = tf.random.normal([num_sample, self.time_steps, self.d_noise_dim])
        
        # Generate data
        if self.conditional:
            # Use static labels if available, otherwise sample random labels
            if self.statics_label is not None:
                # Sample random static labels
                idx = np.random.randint(0, len(self.statics_label), num_sample)
                conditions = tf.convert_to_tensor(self.statics_label[idx], dtype=tf.float32)
            else:
                conditions = tf.random.normal([num_sample, self.num_labels])
                
            c_gen_data = self.c_gan.generator([c_noise, conditions], training=False)
            d_gen_data = self.d_gan.generator([d_noise, conditions], training=False)
        else:
            c_gen_data = self.c_gan.generator(c_noise, training=False)
            d_gen_data = self.d_gan.generator(d_noise, training=False)
            
        # Apply sigmoid to discrete data to ensure 0-1 range
        d_gen_data = tf.sigmoid(d_gen_data)
        
        return d_gen_data.numpy(), c_gen_data.numpy()

    @tf.function
    def vae_loss(self, real_data, reconstructed, sigma, mu, log_sigma):
        """VAE loss function: reconstruction + KL divergence"""
        # Reconstruction loss
        re_loss = tf.reduce_mean(tf.square(real_data - reconstructed))
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(1 + log_sigma - tf.square(mu) - tf.exp(log_sigma))
        
        return re_loss, kl_loss
    
    @tf.function
    def train_vae_step(self, real_data, vae_model, optimizer, conditions=None):
        """Single VAE training step"""
        with tf.GradientTape() as tape:
            # Forward pass
            if conditions is not None:
                reconstructed, sigma, mu, log_sigma, z = vae_model([real_data, conditions], training=True)
            else:
                reconstructed, sigma, mu, log_sigma, z = vae_model(real_data, training=True)
                
            # Calculate loss
            re_loss, kl_loss = self.vae_loss(real_data, reconstructed, sigma, mu, log_sigma)
            total_loss = self.alpha_re * re_loss + self.alpha_kl * kl_loss
            
        # Calculate gradients and apply
        gradients = tape.gradient(total_loss, vae_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae_model.trainable_variables))
        
        return re_loss, kl_loss
    
    @tf.function
    def train_generator_step(self, noise, generator, discriminator, optimizer, conditions=None):
        """Single generator training step"""
        with tf.GradientTape() as tape:
            # Generate fake data
            if conditions is not None:
                fake_data = generator([noise, conditions], training=True)
            else:
                fake_data = generator(noise, training=True)
                
            # Discriminator output on fake data
            fake_output = discriminator(fake_data, training=False)
            
            # Generator loss: fooling the discriminator
            gen_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_output), fake_output))
            
        # Calculate gradients and apply
        gradients = tape.gradient(gen_loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        
        return gen_loss, fake_data
    
    @tf.function
    def train_discriminator_step(self, real_data, fake_data, discriminator, optimizer):
        """Single discriminator training step"""
        with tf.GradientTape() as tape:
            # Discriminator output on real and fake data
            real_output = discriminator(real_data, training=True)
            fake_output = discriminator(fake_data, training=True)
            
            # Real and fake losses
            real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_output), real_output))
            fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.zeros_like(fake_output), fake_output))
            
            # Total discriminator loss
            total_loss = real_loss + fake_loss
            
        # Calculate gradients and apply
        gradients = tape.gradient(total_loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
        
        return total_loss
    
    def train(self):
        """Train the M3GAN model"""
        print("Starting M3GAN training...")
        
        # Create data directories
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        # Restore from checkpoint if available
        latest_checkpoint = self.manager.latest_checkpoint
        if latest_checkpoint:
            print(f"Restoring from checkpoint: {latest_checkpoint}")
            self.checkpoint.restore(latest_checkpoint)
        
        # Prepare training data
        c_data = self.c_data_sample
        d_data = self.d_data_sample
        
        # Convert to TensorFlow tensors if they aren't already
        if not isinstance(c_data, tf.Tensor):
            c_data = tf.convert_to_tensor(c_data, dtype=tf.float32)
        if not isinstance(d_data, tf.Tensor):
            d_data = tf.convert_to_tensor(d_data, dtype=tf.float32)
        
        num_batches = len(c_data) // self.batch_size
        
        print(f"Number of training batches: {num_batches}")
        
        # Lists to track progress
        c_vae_losses = []
        d_vae_losses = []
        c_gen_losses = []
        d_gen_losses = []
        c_dis_losses = []
        d_dis_losses = []
        
        # Pre-training VAEs
        if self.num_pre_epochs > 0:
            print(f"Pre-training VAEs for {self.num_pre_epochs} epochs...")
            for epoch in range(self.num_pre_epochs):
                start_time = time.time()
                
                c_re_loss_total = 0.0
                c_kl_loss_total = 0.0
                d_re_loss_total = 0.0
                d_kl_loss_total = 0.0
                
                # Shuffle data
                idx = np.random.permutation(len(c_data))
                c_data_shuffled = tf.gather(c_data, idx)
                d_data_shuffled = tf.gather(d_data, idx)
                
                if self.conditional:
                    conditions_shuffled = tf.gather(self.statics_label, idx)
                
                # Train on batches
                for batch in range(num_batches):
                    batch_start = batch * self.batch_size
                    batch_end = min((batch + 1) * self.batch_size, len(c_data))
                    
                    c_batch = c_data_shuffled[batch_start:batch_end]
                    d_batch = d_data_shuffled[batch_start:batch_end]
                    
                    if self.conditional:
                        conditions_batch = conditions_shuffled[batch_start:batch_end]
                        # Train continuous VAE
                        c_re_loss, c_kl_loss = self.train_vae_step(
                            c_batch, self.c_vae, self.c_vae_optimizer, conditions_batch)
                        
                        # Train discrete VAE
                        d_re_loss, d_kl_loss = self.train_vae_step(
                            d_batch, self.d_vae, self.d_vae_optimizer, conditions_batch)
                    else:
                        # Train continuous VAE
                        c_re_loss, c_kl_loss = self.train_vae_step(
                            c_batch, self.c_vae, self.c_vae_optimizer)
                        
                        # Train discrete VAE
                        d_re_loss, d_kl_loss = self.train_vae_step(
                            d_batch, self.d_vae, self.d_vae_optimizer)
                    
                    c_re_loss_total += c_re_loss.numpy()
                    c_kl_loss_total += c_kl_loss.numpy()
                    d_re_loss_total += d_re_loss.numpy()
                    d_kl_loss_total += d_kl_loss.numpy()
                
                # Average losses
                c_re_loss_avg = c_re_loss_total / num_batches
                c_kl_loss_avg = c_kl_loss_total / num_batches
                d_re_loss_avg = d_re_loss_total / num_batches
                d_kl_loss_avg = d_kl_loss_total / num_batches
                
                # Track losses
                c_vae_losses.append((c_re_loss_avg, c_kl_loss_avg))
                d_vae_losses.append((d_re_loss_avg, d_kl_loss_avg))
                
                # Print progress
                if epoch % self.epoch_loss_freq == 0 or epoch == self.num_pre_epochs - 1:
                    time_taken = time.time() - start_time
                    print(f"Pre-training epoch {epoch+1}/{self.num_pre_epochs} ({time_taken:.2f}s)")
                    print(f"  C-VAE: RE={c_re_loss_avg:.6f}, KL={c_kl_loss_avg:.6f}")
                    print(f"  D-VAE: RE={d_re_loss_avg:.6f}, KL={d_kl_loss_avg:.6f}")
                
                # Save checkpoint
                if (epoch + 1) % self.epoch_ckpt_freq == 0 or epoch == self.num_pre_epochs - 1:
                    save_path = self.manager.save()
                    print(f"Checkpoint saved at epoch {epoch+1}: {save_path}")
            
            print("Pre-training complete!")
        
        # Main training loop
        print(f"Starting GAN training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Tracking losses for this epoch
            c_gen_loss_total = 0.0
            c_dis_loss_total = 0.0
            d_gen_loss_total = 0.0
            d_dis_loss_total = 0.0
            c_vae_re_loss_total = 0.0
            c_vae_kl_loss_total = 0.0
            d_vae_re_loss_total = 0.0
            d_vae_kl_loss_total = 0.0
            
            # Shuffle data
            idx = np.random.permutation(len(c_data))
            c_data_shuffled = tf.gather(c_data, idx)
            d_data_shuffled = tf.gather(d_data, idx)
            
            if self.conditional:
                conditions_shuffled = tf.gather(self.statics_label, idx)
            
            # Train on batches with progress bar
            for batch in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                batch_start = batch * self.batch_size
                batch_end = min((batch + 1) * self.batch_size, len(c_data))
                
                c_batch = c_data_shuffled[batch_start:batch_end]
                d_batch = d_data_shuffled[batch_start:batch_end]
                
                if self.conditional:
                    conditions_batch = conditions_shuffled[batch_start:batch_end]
                    
                    # Continuous data training
                    # ------------------------
                    
                    # Train VAE
                    for _ in range(self.v_rounds):
                        c_re_loss, c_kl_loss = self.train_vae_step(
                            c_batch, self.c_vae, self.c_vae_optimizer, conditions_batch)
                    
                    c_vae_re_loss_total += c_re_loss.numpy()
                    c_vae_kl_loss_total += c_kl_loss.numpy()
                    
                    # Train discriminator
                    for _ in range(self.d_rounds):
                        # Generate fake data
                        c_noise = tf.random.normal([len(c_batch), self.time_steps, self.c_noise_dim])
                        _, c_fake_batch = self.train_generator_step(
                            c_noise, self.c_gan.generator, self.c_gan.discriminator, 
                            self.c_gen_optimizer, conditions_batch)
                        
                        # Train discriminator on real and fake data
                        c_dis_loss = self.train_discriminator_step(
                            c_batch, c_fake_batch, self.c_gan.discriminator, self.c_dis_optimizer)
                        
                    c_dis_loss_total += c_dis_loss.numpy()
                    
                    # Train generator
                    for _ in range(self.g_rounds):
                        c_noise = tf.random.normal([len(c_batch), self.time_steps, self.c_noise_dim])
                        c_gen_loss, _ = self.train_generator_step(
                            c_noise, self.c_gan.generator, self.c_gan.discriminator, 
                            self.c_gen_optimizer, conditions_batch)
                        
                    c_gen_loss_total += c_gen_loss.numpy()
                    
                    # Discrete data training
                    # ---------------------
                    
                    # Train VAE
                    for _ in range(self.v_rounds):
                        d_re_loss, d_kl_loss = self.train_vae_step(
                            d_batch, self.d_vae, self.d_vae_optimizer, conditions_batch)
                    
                    d_vae_re_loss_total += d_re_loss.numpy()
                    d_vae_kl_loss_total += d_kl_loss.numpy()
                    
                    # Train discriminator
                    for _ in range(self.d_rounds):
                        # Generate fake data
                        d_noise = tf.random.normal([len(d_batch), self.time_steps, self.d_noise_dim])
                        _, d_fake_batch = self.train_generator_step(
                            d_noise, self.d_gan.generator, self.d_gan.discriminator, 
                            self.d_gen_optimizer, conditions_batch)
                        
                        # Train discriminator on real and fake data
                        d_dis_loss = self.train_discriminator_step(
                            d_batch, d_fake_batch, self.d_gan.discriminator, self.d_dis_optimizer)
                        
                    d_dis_loss_total += d_dis_loss.numpy()
                    
                    # Train generator
                    for _ in range(self.g_rounds):
                        d_noise = tf.random.normal([len(d_batch), self.time_steps, self.d_noise_dim])
                        d_gen_loss, _ = self.train_generator_step(
                            d_noise, self.d_gan.generator, self.d_gan.discriminator, 
                            self.d_gen_optimizer, conditions_batch)
                        
                    d_gen_loss_total += d_gen_loss.numpy()
                    
                else:
                    # Unconditional training
                    # Continuous data training
                    # ------------------------
                    
                    # Train VAE
                    for _ in range(self.v_rounds):
                        c_re_loss, c_kl_loss = self.train_vae_step(
                            c_batch, self.c_vae, self.c_vae_optimizer)
                    
                    c_vae_re_loss_total += c_re_loss.numpy()
                    c_vae_kl_loss_total += c_kl_loss.numpy()
                    
                    # Train discriminator
                    for _ in range(self.d_rounds):
                        # Generate fake data
                        c_noise = tf.random.normal([len(c_batch), self.time_steps, self.c_noise_dim])
                        _, c_fake_batch = self.train_generator_step(
                            c_noise, self.c_gan.generator, self.c_gan.discriminator, 
                            self.c_gen_optimizer)
                        
                        # Train discriminator on real and fake data
                        c_dis_loss = self.train_discriminator_step(
                            c_batch, c_fake_batch, self.c_gan.discriminator, self.c_dis_optimizer)
                        
                    c_dis_loss_total += c_dis_loss.numpy()
                    
                    # Train generator
                    for _ in range(self.g_rounds):
                        c_noise = tf.random.normal([len(c_batch), self.time_steps, self.c_noise_dim])
                        c_gen_loss, _ = self.train_generator_step(
                            c_noise, self.c_gan.generator, self.c_gan.discriminator, 
                            self.c_gen_optimizer)
                        
                    c_gen_loss_total += c_gen_loss.numpy()
                    
                    # Discrete data training
                    # ---------------------
                    
                    # Train VAE
                    for _ in range(self.v_rounds):
                        d_re_loss, d_kl_loss = self.train_vae_step(
                            d_batch, self.d_vae, self.d_vae_optimizer)
                    
                    d_vae_re_loss_total += d_re_loss.numpy()
                    d_vae_kl_loss_total += d_kl_loss.numpy()
                    
                    # Train discriminator
                    for _ in range(self.d_rounds):
                        # Generate fake data
                        d_noise = tf.random.normal([len(d_batch), self.time_steps, self.d_noise_dim])
                        _, d_fake_batch = self.train_generator_step(
                            d_noise, self.d_gan.generator, self.d_gan.discriminator, 
                            self.d_gen_optimizer)
                        
                        # Train discriminator on real and fake data
                        d_dis_loss = self.train_discriminator_step(
                            d_batch, d_fake_batch, self.d_gan.discriminator, self.d_dis_optimizer)
                        
                    d_dis_loss_total += d_dis_loss.numpy()
                    
                    # Train generator
                    for _ in range(self.g_rounds):
                        d_noise = tf.random.normal([len(d_batch), self.time_steps, self.d_noise_dim])
                        d_gen_loss, _ = self.train_generator_step(
                            d_noise, self.d_gan.generator, self.d_gan.discriminator, 
                            self.d_gen_optimizer)
                        
                    d_gen_loss_total += d_gen_loss.numpy()
            
            # Average losses
            c_vae_re_loss_avg = c_vae_re_loss_total / num_batches
            c_vae_kl_loss_avg = c_vae_kl_loss_total / num_batches
            c_gen_loss_avg = c_gen_loss_total / num_batches
            c_dis_loss_avg = c_dis_loss_total / num_batches
            
            d_vae_re_loss_avg = d_vae_re_loss_total / num_batches
            d_vae_kl_loss_avg = d_vae_kl_loss_total / num_batches
            d_gen_loss_avg = d_gen_loss_total / num_batches
            d_dis_loss_avg = d_dis_loss_total / num_batches
            
            # Track losses
            c_vae_losses.append((c_vae_re_loss_avg, c_vae_kl_loss_avg))
            d_vae_losses.append((d_vae_re_loss_avg, d_vae_kl_loss_avg))
            c_gen_losses.append(c_gen_loss_avg)
            d_gen_losses.append(d_gen_loss_avg)
            c_dis_losses.append(c_dis_loss_avg)
            d_dis_losses.append(d_dis_loss_avg)
            
            # Print progress
            if epoch % self.epoch_loss_freq == 0 or epoch == self.num_epochs - 1:
                time_taken = time.time() - start_time
                print(f"Epoch {epoch+1}/{self.num_epochs} ({time_taken:.2f}s)")
                print(f"  C-VAE: RE={c_vae_re_loss_avg:.6f}, KL={c_vae_kl_loss_avg:.6f}")
                print(f"  C-GAN: G={c_gen_loss_avg:.6f}, D={c_dis_loss_avg:.6f}")
                print(f"  D-VAE: RE={d_vae_re_loss_avg:.6f}, KL={d_vae_kl_loss_avg:.6f}")
                print(f"  D-GAN: G={d_gen_loss_avg:.6f}, D={d_dis_loss_avg:.6f}")
                
                # Generate samples at intervals
                if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                    print("Generating samples...")
                    d_samples, c_samples = self.generate_data(num_sample=10)
                    
                    # Save samples if desired
                    sample_dir = os.path.join(self.checkpoint_dir, f"epoch{epoch+1}")
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    np.savez(os.path.join(sample_dir, "gen_data.npz"), 
                              d_gen_data=d_samples, c_gen_data=c_samples)
            
            # Save checkpoint
            if (epoch + 1) % self.epoch_ckpt_freq == 0 or epoch == self.num_epochs - 1:
                save_path = self.manager.save()
                print(f"Checkpoint saved at epoch {epoch+1}: {save_path}")
        
        print("Training complete!")
        self.plot_training_curves(c_vae_losses, d_vae_losses, c_gen_losses, d_gen_losses, c_dis_losses, d_dis_losses)
    
    def plot_training_curves(self, c_vae_losses, d_vae_losses, c_gen_losses, d_gen_losses, c_dis_losses, d_dis_losses):
        """Plot training curves for VAE and GAN losses"""
        epochs = range(1, len(c_vae_losses) + 1)
        
        # Plot VAE losses
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, [loss[0] for loss in c_vae_losses], label='C-VAE RE Loss')
        plt.plot(epochs, [loss[1] for loss in c_vae_losses], label='C-VAE KL Loss')
        plt.plot(epochs, [loss[0] for loss in d_vae_losses], label='D-VAE RE Loss')
        plt.plot(epochs, [loss[1] for loss in d_vae_losses], label='D-VAE KL Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('VAE Losses')
        
        # Plot GAN losses
        plt.subplot(1, 2, 2)
        plt.plot(epochs, c_gen_losses, label='C-GAN Generator Loss')
        plt.plot(epochs, c_dis_losses, label='C-GAN Discriminator Loss')
        plt.plot(epochs, d_gen_losses, label='D-GAN Generator Loss')
        plt.plot(epochs, d_dis_losses, label='D-GAN Discriminator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Losses')
        
        plt.tight_layout()
        plt.show()
    