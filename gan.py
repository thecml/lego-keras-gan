import numpy as np
from tensorflow import keras
import os

BASE_DIR = 'PATH_TO_BASE_DIR'
N_CHANNELS = 3
LATENT_DIM = 100
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 32

def load_lego_data():
    data_generator = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras.applications.xception.preprocess_input,
        data_format='channels_last',
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False)

    flow_from_directory_params = {'target_size': (IMAGE_HEIGHT, IMAGE_WIDTH),
                                'color_mode': 'grayscale' if N_CHANNELS == 1 else 'rgb',
                                'class_mode': None,
                                'batch_size': BATCH_SIZE}

    image_generator = data_generator.flow_from_directory(
        directory=os.path.join(BASE_DIR, 'training_data'),
        **flow_from_directory_params)
    
    return image_generator

def get_image_batch(image_generator):
    img_batch = image_generator.next()

    if len(img_batch) != BATCH_SIZE:
        img_batch = image_generator.next()

    assert img_batch.shape == (BATCH_SIZE, IMAGE_HEIGHT,
     IMAGE_WIDTH, N_CHANNELS), img_batch.shape
    return img_batch

def make_generator():
    generator = keras.models.Sequential()
    # Foundation for 4x4 image
    generator.add(keras.layers.Dense(256*4*4, input_dim=LATENT_DIM))
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    generator.add(keras.layers.Reshape((4, 4, 256)))
    # Upsample to 8x8
    generator.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    # Upsample to 16x16
    generator.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    # Upsample to 32x32
    generator.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    # Upsample to 64x64
    generator.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    # Upsample to 128x128
    generator.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    # Output 128x128
    generator.add(keras.layers.Conv2D(N_CHANNELS, (3,3), activation='tanh', padding='same'))
    return generator

def make_discriminator():
    discriminator = keras.models.Sequential()
    # Normal
    discriminator.add(keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)))
    discriminator.add(keras.layers.LeakyReLU(alpha=0.2))
    # Downsample to 64x64
    discriminator.add(keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same'))
    discriminator.add(keras.layers.LeakyReLU(alpha=0.2))
    # Downsample to 32x32
    discriminator.add(keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same'))
    discriminator.add(keras.layers.LeakyReLU(alpha=0.2))
    # Downsample to 16x16
    discriminator.add(keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same'))
    discriminator.add(keras.layers.LeakyReLU(alpha=0.2))
    # Downsample to 8x8
    discriminator.add(keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same'))
    discriminator.add(keras.layers.LeakyReLU(alpha=0.2))
    # Classifier
    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dropout(0.4))
    discriminator.add(keras.layers.Dense(1, activation='sigmoid'))
    opt = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
    discriminator.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return discriminator

def train_gan(gan, generator, discriminator, image_generator, n_epochs=1000):
    # Run training loop for n_epochs
    for epoch in range(n_epochs):
        real_images = get_image_batch(image_generator)
        random_index = np.random.randint(0, real_images.shape[0], BATCH_SIZE)
        real_data = real_images[random_index]
        real_labels = np.ones((BATCH_SIZE, 1))
        d_loss1, _ = discriminator.train_on_batch(real_data, real_labels)
        
        random_latent_vectors = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
        fake_data = generator.predict(random_latent_vectors)
        fake_labels = np.zeros((BATCH_SIZE, 1))
        d_loss2, _ = discriminator.train_on_batch(fake_data, fake_labels)
        
        random_latent_vectors = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
        misleading_labels = np.ones((BATCH_SIZE, 1))
        g_loss = gan.train_on_batch(random_latent_vectors, misleading_labels)
        
        print('Epoch %d, d1=%.3f, d2=%.3f g=%.3f' % (epoch+1, d_loss1, d_loss2, g_loss))
        
        # Save GAN weights, a fake and a real image every 50 epochs
        if (epoch+1) % 500 == 0:
            gan.save_weights(os.path.join(BASE_DIR, 'gan\\gan.h5'))
            print(f'Saving batch of generated images at adversarial step: {epoch}.')
            random_latent_vectors = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
            fake_images = generator.predict(random_latent_vectors)
            real_images = get_image_batch(image_generator)
            save_dir = os.path.join(BASE_DIR, 'gan')
            img = keras.preprocessing.image.array_to_img(fake_images[0])
            img.save(os.path.join(save_dir, 'generated_lego_' + str(epoch) + '.png'))
            img = keras.preprocessing.image.array_to_img(real_images[0])
            img.save(os.path.join(save_dir, 'real_lego_' + str(epoch) + '.png'))
        
def main():
    # Make the GAN
    generator = make_generator()
    discriminator = make_discriminator()
    discriminator.trainable = False
    gan = keras.models.Sequential([generator, discriminator])

    # Compile the GAN
    opt = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=opt, loss='binary_crossentropy')

    # Initialize the data generator
    image_generator = load_lego_data()

    # Train the GAN
    train_gan(gan, generator, discriminator, image_generator)

if __name__ == "__main__":
    main()