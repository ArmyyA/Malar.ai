import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Dense

# Read and resize the images
images = []
labels = []
positive_dir = 'malaria-dataset/positive'
negative_dir = 'malaria-dataset/negative'

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
X_train, X_val, y_train, y_val = train_test_split(gray_images, labels, test_size=0.2, random_state=42)


# Load the pre-trained ResNet50 model
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global spatial average pooling layer
x = resnet.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Add a logistic layer
predictions = Dense(1, activation='sigmoid')(x)

# Create a new model
model = Model(inputs=resnet.input, outputs=predictions)

# Get the output tensor of the last layer in the model
last_layer_output = model.output

# Add a dense layer
x = Dense(2, activation='softmax')(last_layer_output)

# Create a new model with the new output tensor
model = Model(inputs=model.input, outputs=x)

# Freeze the layers of the pre-trained model
for layer in resnet.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Train the model
def preprocess_data(X_train, y_train, X_val, y_val):
    X_train = np.stack(X_train)
    X_val = np.stack(X_val)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = preprocess_data(X_train, y_train, X_val, y_val)
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)

X_train = X_train / 255
X_val = X_val / 255

# Change the input data to have 3 channels
X_train = np.repeat(X_train, 3, axis=-1)
X_val = np.repeat(X_val, 3, axis=-1)

# Train the model
print("Model Training will now begin:")
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

print("Training complete!")

# Evaluate the Model
results = model.evaluate(X_val, y_val, batch_size=32)
print(f'Loss: {results[0]}, Accuracy: {results[1]}')

# Save the Model
model.save('malaria_detection_model.h5')
print("Model saved!")

