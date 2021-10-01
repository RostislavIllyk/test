
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


"""
## Load the data: the Cats vs Dogs dataset

curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
"""
"""
## Generate a `Dataset`
"""
IMG_SIZE = 180
image_size = (180, 180)
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255.0)
    return image, label


def augment(image_label, seed):
    image, label = image_label
    image, label = resize_and_rescale(image, label)

    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    image = tf.image.random_flip_left_right(image)
    image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
    image = tf.clip_by_value(image, 0, 1)
    return image, label


# Create a generator
rng = tf.random.Generator.from_seed(123, alg='philox')


# A wrapper function for updating seeds
def f(x, y):
    seed = rng.make_seeds(2)[0]
    image, label = augment((x, y), seed)
    return image, label


train_ds = (
    train_ds
    .shuffle(1000)
    .map(f, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    #x = data_augmentation(inputs)

    # Entry block
    #x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)


"""
## Train the model
"""

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=7,
                                                  mode='min',
                                                  restore_best_weights=True)


model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=[early_stopping], validation_data=val_ds,
)


"""
## Test the model
"""
test_img_path = "PetImages/Cat/6779.jpg"

img = tf.io.read_file(test_img_path)
img = tf.image.decode_jpeg(img)

img = tf.image.resize(
    img, [180, 180], method=tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio=False,
    antialias=False, name=None
)

img_array = keras.preprocessing.image.img_to_array(img)/255
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)


print('All base done...')

model.save('base')
print('base saved!')
