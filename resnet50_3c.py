import tensorflow as tf
import keras
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    "Data/train",
    color_mode='grayscale',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "Data/valid",
    color_mode='grayscale',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "Data/test",
    color_mode='grayscale',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Store class names
class_names = train_ds.class_names
print("Classes:", class_names)

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))




# Function to repeat grayscale to 3 channels
def to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

train_ds = train_ds.map(lambda x, y: (to_rgb(x), y))
val_ds = val_ds.map(lambda x, y: (to_rgb(x), y))
test_ds = test_ds.map(lambda x, y: (to_rgb(x), y))

# Build ResNet50 model
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base layers
base_model.trainable = False

model = keras.Sequential([
    #base_model,
    keras.Input(shape=(None, None, 3)),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.2f}") #0.38
