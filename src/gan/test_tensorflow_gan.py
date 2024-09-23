import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
from PIL import Image

# Load and preprocess dataset
def load_images(image_dir, image_size):
    images = []
    for directory in os.listdir(image_dir):
        for filename in os.listdir(os.path.join(image_dir, directory)):
            img = Image.open(os.path.join(image_dir, directory, filename))
            if img is not None:
                img = img.resize(image_size)
                img = np.array(img)
                img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
                images.append(img)
    return np.array(images)

image_dir = '../data/images_jpg'  # Update this to the actual path of your images
image_size = (3072, 2048)  # Target size for images
dataset = load_images(image_dir, image_size)

# Define the GAN
def build_generator():
    model = Sequential()
    model.add(Dense(256*8*8, activation="relu", input_dim=100))
    model.add(Reshape((8, 8, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation='tanh'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=(3072, 2048, 3)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Compile GAN
gan_input = tf.keras.Input(shape=(100,))
generated_image = generator(gan_input)
discriminator.trainable = False
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

checkpoint_dir = 'results/gan_tf'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'gan_weights_epoch_{epoch:02d}.hdf5'),
    save_weights_only=True,
    save_freq='epoch',
    period=100  # Save every 100 epochs
)

# Training the GAN
import tqdm

def train_gan(gan, generator, discriminator, dataset, epochs, batch_size, checkpoint_callback):
    for epoch in range(epochs):
        for _ in range(batch_size):
            # Train discriminator
            noise = np.random.normal(0, 1, (batch_size, 100))
            generated_images = generator.predict(noise)
            real_images = dataset[np.random.randint(0, dataset.shape[0], batch_size)]
            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(real_images, labels_real)
            d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            labels = np.ones((batch_size, 1))
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, labels)
        
        print(f'Epoch {epoch+1}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')

        # Save checkpoints every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_callback.on_epoch_end(epoch, logs={'loss': g_loss})

# Parameters
epochs = 10000
batch_size = 32

# Train GAN
train_gan(gan, generator, discriminator, dataset, epochs, batch_size)

# Generate synthetic images
def generate_images(generator, num_images, noise_dim=100):
    noise = np.random.normal(0, 1, (num_images, noise_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]
    return generated_images

synthetic_images = generate_images(generator, 10)

# Save generated images
output_dir = 'figs/gen_imgs_tf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for i, img in enumerate(synthetic_images):
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(output_dir, f'synthetic_image_{i}.png'))
