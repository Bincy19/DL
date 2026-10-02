import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping

# Set the base directory
base_dir = "/home/anbi23bd/DRUMSTER/DRUMSTER"
print("Contents of the base directory:")
print(os.listdir(base_dir))
num_skipped = 0

# Loop through "DAMAGED" and "UNDAMAGED" folders
for folder_name in ("DAMAGED", "UNDAMAGED"):
    folder_path = os.path.join(base_dir, folder_name)

    # Iterate over files in the folder
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)

        # Skip if it's a directory
        if os.path.isdir(fpath):
            continue

        try:
            # Open the file in binary mode
            with open(fpath, "rb") as fobj:
                # Read the first 10 bytes of the file and check for "JFIF" string
                is_jfif = b"JFIF" in fobj.peek(10)
        except Exception as e:
            print(f"Error reading file {fpath}: {e}")
            continue

        # If "JFIF" is not found, delete the corrupted image
        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")

# Load the dataset
img_height, img_width = 180, 180
batch_size = 32

train_ds = image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode="int",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode="int",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Prefetching data to optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Build the ResNet50 model
resnet_model = Sequential()
pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(180, 180, 3),
                                                  pooling='avg',
                                                  weights='imagenet')
for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(1, activation='sigmoid'))

resnet_model.summary()

# Adjust the learning rate
resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

# Increase patience for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Train the model with early stopping
epochs = 25
history = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Evaluate the model
print("\nThe results for DAMAGED vs UNDAMAGED Dataset using ResNet")
val_loss, val_accuracy = resnet_model.evaluate(val_ds)
print(f"Validation Loss For DAMAGED vs UNDAMAGED Dataset using ResNet is: {val_loss:.4f}")
print(f"Validation Accuracy For DAMAGED vs UNDAMAGED Dataset using ResNet is: {val_accuracy:.4f}")

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Generate predictions on the validation dataset
y_pred_prob = resnet_model.predict(val_ds)
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

# Plot training history
history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

epochs_range = range(1, len(train_loss) + 1)
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
plt.savefig('training_validation_plots(resnet).jpg', format='jpg')
plt.show()
