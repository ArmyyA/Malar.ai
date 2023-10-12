import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D
from keras.models import Model

# Read and resize the images
images = []
labels = []
positive_dir = 'images/Parasitized'
negative_dir = 'images/Uninfected'

for filename in os.listdir(positive_dir):
    print('first')
    print(filename)
    img = cv2.imread(os.path.join(positive_dir, filename))
    if img is None:
        print("image not found or corrupted")
    else:
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(1)

for filename in os.listdir(negative_dir):
    print(filename)
    print('second')

    if not filename.startswith('.'):
        img = cv2.imread(os.path.join(negative_dir, filename))
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(0)

# Convert the images to grayscale
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Split the data into training and test sets
X_train, X_val, y_train, y_val = train_test_split(
    gray_images, labels, test_size=0.2, random_state=42)

# Custom input layer for the grayscale images
input_tensor = Input(shape=(224, 224, 1))

# Load the ResNet50 model
resnet = ResNet50(weights='imagenet', include_top=False,
                  input_shape=(224, 224, 3))

conv1_weights = resnet.layers[1].getweights()
gray_weights = [0.299 * conv1_weights[0][:, :, 0, :] + 0.587 *
                conv1_weights[0][:, :, 1, :] + 0.114 * conv1_weights[0][:, :, 2, :]]
gray_weights.append(conv1_weights[1])

# Create a new convolutional layer for grayscale and set its weights
new_conv1 = Conv2D(64, (7, 7), strides=(
    2, 2), padding='valid', name='conv1')(input_tensor)
new_conv1_layer = Model(inputs=input_tensor, outputs=new_conv1)
new_conv1_layer.layers[1].set_weights(gray_weights)

# Reconstruct ResNet with the grayscale layer
custom_resnet_output = new_conv1
for layer in resnet.layers[2:]:
    custom_resnet_output = layer(custom_resnet_output)

# Add a global spatial average pooling layer
x = GlobalAveragePooling2D()(custom_resnet_output)

# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Add a logistic layer for binary classification
predictions = Dense(1, activation='sigmoid')(x)

# Create a new model
model = Model(inputs=input_tensor, outputs=predictions)

# Freeze the layers in modified ResNet50
for layer in new_conv1_layer.layers + resnet.layers:
    layer.trainable = False

# Compile the model using binary_crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])


# Define preprocessing method using np stack and array
def preprocess_data(X_train, y_train, X_val, y_val):
    X_train = np.stack(X_train)
    X_val = np.stack(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    return X_train, y_train, X_val, y_val


# Preprocess the data
X_train, y_train, X_val, y_val = preprocess_data(
    X_train, y_train, X_val, y_val)
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)


# Change the input data to have 3 channels
X_train = np.repeat(X_train, 3, axis=-1)
X_val = np.repeat(X_val, 3, axis=-1)

# ImageNet mean
mean_value = (123.68 + 116.779 + 103.939) / 3.0

X_train = (X_train - mean_value) / 255.0
X_val = (X_val - mean_value) / 255.0

# Train the model
print("Model Training will now begin:")
history = model.fit(X_train, y_train, batch_size=32,
                    epochs=10, validation_data=(X_val, y_val))

print("Training complete!")

# Evaluate the Model
results = model.evaluate(X_val, y_val, batch_size=32)
print(f'Loss: {results[0]}, Accuracy: {results[1]}')

# Save the Model
model.save('malaria_detection_model.h5')
print("Model saved!")
