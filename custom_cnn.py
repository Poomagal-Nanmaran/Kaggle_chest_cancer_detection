import tensorflow as tf
#from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/train",
    color_mode='grayscale',     # <- keep grayscale
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
cls_names=train_ds.class_names
print(f"Number of classes: {len(train_ds.class_names)}")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/valid",
    color_mode='grayscale',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/test",
    color_mode='grayscale',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255, offset=0.0)#tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

def build_grayscale_cnn(input_shape=(224, 224, 1), num_classes=4):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_grayscale_cnn(num_classes=len(cls_names))
model.summary()
# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)
# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.2f}") #0.37
# Save the model
#model.save("custom_cnn_grayscale.h5")   
