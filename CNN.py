#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import data as tf_data
from sklearn.metrics import recall_score

# Step 1: Clean the dataset
location = "/home/anbi23bd/DRUMSTER/DRUMSTER"
num_skipped = 0
for folder_name in ("DAMAGED", "UNDAMAGED"):
    folder_path = os.path.join(location, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")

# Step 2: Load and prepare the dataset
image_size = (180, 180)
batch_size = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    location,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    location,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Step 3: Data augmentation
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images, training=False):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x), y))

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

# Step 4: Define the model
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
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
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=2)

print('Training DAMAGED vs UNDAMAGED dataset - \n')
print("Learning Rate: 0.0001, Batch Size=64, Epoch=25")

# Step 5: Compile and train the model
epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# Step 6: Evaluate the model
print("The results for DAMAGED vs UNDAMAGED Dataset")
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Loss For DAMAGED vs UNDAMAGED Dataset is: {val_loss}")
print(f"Validation Accuracy For DAMAGED vs UNDAMAGED Dataset is: {val_accuracy}")

#Step 7
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Generate predictions on the validation dataset
y_pred_prob = model.predict(val_ds)
y_pred = np.where(y_pred_prob > 0.5, 1, 0).flatten()  # Convert probabilities to binary labels

# Extract true labels from the validation dataset
y_true = np.concatenate([y for x, y in val_ds], axis=0)

# Convert y_true to binary labels if needed
y_true_binary = np.where(y_true > 0, 1, 0)

# Make sure y_pred and y_true_binary have the same length
min_length = min(len(y_pred), len(y_true_binary))
y_pred = y_pred[:min_length]
y_true_binary = y_true_binary[:min_length]

# Calculate precision, recall, and F1 score
precision = precision_score(y_true_binary, y_pred)
recall = recall_score(y_true_binary, y_pred)
f1 = f1_score(y_true_binary, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Step 8: Plot and save training history
history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_accuracy = history_dict['acc']
val_accuracy = history_dict['val_acc']

epochs_range = range(1, epochs + 1)
plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, 'b', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracy, 'b', label='Training accuracy')
plt.plot(epochs_range, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Save the plot as a JPEG image
plt.savefig('training_validation_plots_cnn.jpg', format='jpg')
plt.show()
