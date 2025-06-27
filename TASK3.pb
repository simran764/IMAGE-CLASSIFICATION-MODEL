import tensorflow as tf
from tensorflow.keras import layers, models, utils
import matplotlib.pyplot as plt
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")

# --- 1. Load and Preprocess the MNIST Dataset ---
print("Loading MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Reshape images to add a channel dimension (for grayscale, it's 1 channel)
# CNNs expect input in the format (batch_size, height, width, channels)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values from [0, 255] to [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot encode the labels (e.g., 3 -> [0,0,0,1,0,0,0,0,0,0])
# There are 10 classes (digits 0-9)
train_labels = utils.to_categorical(train_labels, num_classes=10)
test_labels = utils.to_categorical(test_labels, num_classes=10)

print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}\n")

# --- 2. Define the CNN Model Architecture ---
print("Defining CNN model architecture...")
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third Convolutional Block (Optional, but adds more learning capacity)
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten the 2D feature maps to a 1D vector for Dense layers
    layers.Flatten(),

    # Dense (Fully Connected) Layers
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # Output layer: 10 classes, softmax for probabilities
])

# Display the model summary to see the layers and parameter counts
model.summary()
print("\n")

# --- 3. Compile the Model ---
print("Compiling the model...")
model.compile(optimizer='adam', # Adam optimizer is a good default choice
              loss='categorical_crossentropy', # Appropriate loss for multi-class classification with one-hot labels
              metrics=['accuracy']) # Track accuracy during training and evaluation
print("Model compiled successfully.\n")

# --- 4. Train the Model ---
print("Training the model...")
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)
# epochs: Number of times to iterate over the entire training dataset
# batch_size: Number of samples per gradient update
# validation_split: Reserve 10% of training data for validation during training
print("Model training complete.\n")

# --- 5. Evaluate the Model ---
print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}\n")

# --- 6. Visualize Training History (Optional but Recommended) ---
# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plot_history_path = 'cnn_training_history.png'
plt.savefig(plot_history_path)
plt.close() # Close the plot to free memory
print(f"Training history plot saved to '{plot_history_path}'")

# --- Optional: Make a prediction on a single image ---
# Select a random image from the test set
random_index = np.random.randint(0, len(test_images))
sample_image = test_images[random_index]
true_label = np.argmax(test_labels[random_index])

# The model expects a batch of images, so add an extra dimension
sample_image_batch = np.expand_dims(sample_image, axis=0)

# Make a prediction
predictions = model.predict(sample_image_batch)
predicted_label = np.argmax(predictions[0]) # Get the index of the highest probability

print(f"\nPrediction for a random test image (Index: {random_index}):")
print(f"True Label: {true_label}")
print(f"Predicted Label: {predicted_label}")
print(f"Prediction Probabilities: {np.round(predictions[0], 2)}")

# Visualize the sample image
plt.figure(figsize=(4, 4))
plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f"True: {true_label}, Predicted: {predicted_label}")
plt.axis('off')
prediction_image_path = 'cnn_sample_prediction.png'
plt.savefig(prediction_image_path)
plt.close()
print(f"Sample prediction image saved to '{prediction_image_path}'")
