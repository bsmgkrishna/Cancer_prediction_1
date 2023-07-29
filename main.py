from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO


def initiateGenerator(path):
    base_path = path

    print("\nTotal : ", end=" ")
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(batch_size=32, directory=base_path)
    train_datagen = ImageDataGenerator(validation_split=0.2)
    print("\nFor Training : ", end=" ")
    train_generator = train_datagen.flow_from_directory(
        base_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training')
    print("\nFor Val : ", end=" ")
    validation_generator = train_datagen.flow_from_directory(
        base_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation', shuffle=False)

    # Get the class names and number of classes in the dataset
    class_names = train_dataset.class_names
    noOfClasses = len(class_names)
    print("\nNo of Classes : ", noOfClasses)
    print("Classes : ", class_names)
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(noOfClasses):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    for image_batch, labels_batch in train_dataset:
        print("Image Shape : ", image_batch.shape)
        break
    return noOfClasses, class_names, train_generator, validation_generator


def initiateNormalize():
    """
    Preprocesses and normalizes the training and validation datasets.

    Returns:
        train_ds (tf.data.Dataset): Normalized training dataset.
        val_ds (tf.data.Dataset): Normalized validation dataset.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    # Cache, shuffle, and prefetch the training dataset
    train_ds = train_generator.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    # Cache and prefetch the validation dataset
    val_ds = val_generator.cache().prefetch(buffer_size=AUTOTUNE)

    # Create a Rescaling layer to normalize the pixel values between 0 and 1
    normalization_layer = layers.Rescaling(1. / 255)

    # Apply the normalization layer to the training dataset
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    # Get a batch of normalized images and labels
    image_batch, labels_batch = next(iter(normalized_ds))

    # Retrieve the first image from the batch and print its minimum and maximum pixel values
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))

    return train_ds, val_ds


def initiateModel(noOfClasses):
    """
    Initializes the MobileNetV3 model with pre-trained ImageNet weights for multi-class classification

    Args:
        noOfClasses (int): Number of classes for the classification task.

    Returns:
        tf.keras.Model: Initialized MobileNetV3 model.
    """
    # Initialize the MobileNetV3 model with pre-trained ImageNet weights
    modelInput = tf.keras.applications.MobileNetV3Small(
        input_shape=IMAGE_SIZE + [3],
        include_top=False,
        weights="imagenet"
    )

    # Set all the layers in the MobileNetV3 model as non-trainable
    for layer in modelInput.layers:
        layer.trainable = False

    # Flatten the output of the MobileNetV3 model
    x = Flatten()(modelInput.output)

    # Add a fully connected softmax layer for the classification task
    prediction = Dense(noOfClasses, activation='softmax')(x)

    # Create the final model by combining the MobileNetV3 base model and the prediction layer
    model = Model(inputs=modelInput.input, outputs=prediction)

    return model


def modelSummary(model):
    """
    Prints the summary of the model, including the architecture and number of parameters.

    Args:
        model (tf.keras.Model): Model to print the summary of.
    """
    # Print the summary of the model, including the architecture and number of parameters
    model.summary()




def initiateParams(className, model, lr):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile the model with the optimizer, loss function, and evaluation metric
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Create a ReduceLROnPlateau callback to reduce the learning rate when validation accuracy plateaus
    annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)

    # Create a ModelCheckpoint callback to save the best model during training
    checkpoint = ModelCheckpoint(className + 'MobileNet.h5', verbose=1, save_best_only=True)

    # Return the model and the callbacks
    return model, annealer, checkpoint


def modelFit(model, annealer, checkpoint, epochs=20, batchSize=256):
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        batch_size=batchSize,
        callbacks=[annealer, checkpoint],
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator)
    )

    return history


def plotOutput(history, className, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(3, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    plt.savefig(className + '_graph.png')


def evalModel(model):
    evl = model.evaluate(validation_generator)
    acc = evl[1] * 100
    msg = f'Accuracy on the Test Set = {acc:5.2f} %'
    print(msg)


def saveModel(model, className):
    model.save(className + " - MobileNetV3.h5")
    print("Model Saved!")



def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.savefig(title + '.png')

def callPlot(model, className, classes):
    y_true = validation_generator.classes
    print("True : ", (y_true))
    y_pred = model.predict(validation_generator)
    y_pred = np.argmax(y_pred, axis=1)
    print("Predicted : ", (y_pred))
    conf_mat = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm=conf_mat,
                          normalize=False,
                          target_names=classes,
                          title=className + " Confusion Matrix")


def load_image(image_path_or_url):
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    return image


# ... (previous code remains the same)

def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert('RGB')  # Convert to RGB mode for PNG images
    image = np.array(image)   # Normalize the pixel values between 0 and 1
    image = np.expand_dims(image, axis=0)
    return image



def predict_cancer_type(image_path_or_url):
    # Load the image
    image = load_image(image_path_or_url)

    # Preprocess the image
    image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    # Check if the predicted class is in the list of class names from the training dataset
    if predicted_class in class_names:
        # Load the image for visualization
        image_for_display = cv2.cvtColor(cv2.imread(image_path_or_url), cv2.COLOR_BGR2RGB)

        # Get the probability score for the predicted class
        predicted_prob = predictions[0][predicted_class_index]

        # Prepare the text to be displayed on the image
        text = f"Predicted: {predicted_class}\nProbability: {predicted_prob:5.3f}"

        # Resize the image and create a blank canvas for the enlarged image
        enlarged_image = cv2.resize(image_for_display, (800, 800))
        canvas = np.ones_like(enlarged_image) * 255

        # Calculate the position for the text on the canvas
        text_position = (20, 40)

        # Paste the enlarged image onto the canvas
        canvas[:enlarged_image.shape[0], :enlarged_image.shape[1], :] = enlarged_image

        # Display the image with the predicted class and probability
        plt.imshow(canvas)
        plt.title("Cancer Type Prediction")
        plt.text(*text_position, text, fontsize=16, color='black', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        plt.axis("off")
        plt.show()

        return predicted_class
    else:
        print("Wrong Input: The provided image does not belong to any of the trained classes.")
        return "Wrong Input"


mpath = r'C:\Users\Loges\Downloads\archive (2)\Multi Cancer'
classPaths = os.listdir(mpath)
IMAGE_SIZE = [224, 224]
img_height = 224
img_width = 224
noOfClasses = 0
gEpochs = 1
lr = 0.001
classPaths.sort()
for className in classPaths:
    print(className)

className='ALL'
cpath = os.path.join(mpath, className)
noOfClasses, class_names, train_generator, validation_generator = initiateGenerator(cpath)
curModel = initiateModel(noOfClasses)
modelSummary(curModel)
curModel, annealer, checkpoint = initiateParams(className, curModel, lr)


className='Brain Cancer'

cpath = os.path.join(mpath, className)

# Initiate the generator
noOfClasses, class_names, train_generator, validation_generator = initiateGenerator(cpath)

# Initiate the model
curModel = initiateModel(noOfClasses)

# Print the model summary
modelSummary(curModel)

# Initiate model parameters and callbacks
curModel, annealer, checkpoint = initiateParams(className, curModel, lr)

curHistory = modelFit(curModel, annealer, checkpoint, epochs=gEpochs, batchSize = 256)

# Plot training history
plotOutput(curHistory, className, gEpochs)

# Evaluate the model
evalModel(curModel)

# Save the model
saveModel(curModel, className)

# Plot the confusion matrix
callPlot(curModel, className, class_names)
#
# className='Kidney Cancer'
#
# cpath = os.path.join(mpath, className)
#
# # Initiate the generator
# noOfClasses, class_names, train_generator, validation_generator = initiateGenerator(cpath)
#
# # Initiate the model
# curModel = initiateModel(noOfClasses)
#
# # Print the model summary
# modelSummary(curModel)
#
# # Initiate model parameters and callbacks
# curModel, annealer, checkpoint = initiateParams(className, curModel, lr)
#
# curHistory = modelFit(curModel, annealer, checkpoint, epochs=gEpochs, batchSize = 256)
#
# # Plot training history
# plotOutput(curHistory, className, gEpochs)
#
# # Evaluate the model
# evalModel(curModel)
#
# # Save the model
# saveModel(curModel, className)
#
# # Plot the confusion matrix
# callPlot(curModel, className, class_names)
#
# className='Lung and Colon Cancer'
#
# cpath = os.path.join(mpath, className)
#
# # Initiate the generator
# noOfClasses, class_names, train_generator, validation_generator = initiateGenerator(cpath)
#
# # Initiate the model
# curModel = initiateModel(noOfClasses)
#
# # Print the model summary
# modelSummary(curModel)
#
# # Initiate model parameters and callbacks
# curModel, annealer, checkpoint = initiateParams(className, curModel, lr)
#
# # Train the model
# curHistory = modelFit(curModel, annealer, checkpoint, epochs=gEpochs, batchSize = 256)
#
# # Plot training history
# plotOutput(curHistory, className, gEpochs)
#
# # Evaluate the model
# evalModel(curModel)
#
# # Save the model
# saveModel(curModel, className)
#
# # Plot the confusion matrix
# callPlot(curModel, className, class_names)

# className='Oral Cancer'
#
# cpath = os.path.join(mpath, className)
#
# # Initiate the generator
# noOfClasses, class_names, train_generator, validation_generator = initiateGenerator(cpath)
#
# # Initiate the model
# curModel = initiateModel(noOfClasses)
#
# # Print the model summary
# modelSummary(curModel)
#
# # Initiate model parameters and callbacks
# curModel, annealer, checkpoint = initiateParams(className, curModel, lr)
#
# curHistory = modelFit(curModel, annealer, checkpoint, epochs=gEpochs, batchSize = 256)
#
# # Plot training history
# plotOutput(curHistory, className, gEpochs)
#
# # Evaluate the model
# evalModel(curModel)
#
# # Save the model
# saveModel(curModel, className)
#
# # Plot the confusion matrix
# # callPlot(curModel, className, class_names)

model = tf.keras.models.load_model(className+" - MobileNetV3.h5")
# input_image_path = r"C:\Users\Loges\Downloads\archive (2)\Multi Cancer\Brain Cancer\brain_tumor\brain_tumor_0015.jpg"
# predicted_cancer_type = predict_cancer_type(input_image_path)
# print("Predicted Cancer Type:", predicted_cancer_type)
#
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# import os
#
# # Existing code...
#
# # Create a Tkinter window
# root = tk.Tk()
# root.title("Multi Cancer Diagnosis")
#
# # Function to browse for an image file
# def browse_image():
#     file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
#     if file_path:
#         image_display(file_path)
#
# # Function to display the selected image
# def image_display(image_path):
#     img = Image.open(image_path)
#     img.thumbnail((300, 300))
#     photo = ImageTk.PhotoImage(img)
#     img_label.config(image=photo)
#     img_label.image = photo
#     predicted_cancer_type = predict_cancer_type(image_path)
#     result_label.config(text="Predicted Cancer Type: " + predicted_cancer_type)
#
# # Create browse button
# browse_button = tk.Button(root, text="Browse", command=browse_image)
# browse_button.pack(pady=10)
#
# # Create image label to display the selected image
# img_label = tk.Label(root)
# img_label.pack(pady=10)
#
# # Create label to display the predicted cancer type
# result_label = tk.Label(root, text="")
# result_label.pack(pady=5)
#
# # Run the Tkinter main loop
# root.mainloop()

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# Load the pre-trained MobileNetV3 model
model = tf.keras.models.load_model("Brain Cancer - MobileNetV3.h5")

# Class names for the multicancer diagnosis model
class_names = ["ALL", "Brain Cancer", ...]  # Add other cancer types based on your model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Load the image
            img = Image.open(file)
            img = img.resize((224, 224))
            img = img.convert('RGB')
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Make the prediction using the model
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index]
            predicted_prob = predictions[0][predicted_class_index]

            # Prepare the image for display
            image_for_display = cv2.cvtColor(np.array(img) * 255, cv2.COLOR_RGB2BGR)
            enlarged_image = cv2.resize(image_for_display, (800, 800))
            canvas = np.ones_like(enlarged_image) * 255
            text = f"Predicted: {predicted_class}\nProbability: {predicted_prob:5.3f}"
            text_position = (20, 40)
            canvas[:enlarged_image.shape[0], :enlarged_image.shape[1], :] = enlarged_image

            # Save the output image for display
            output_image_path = "static/output_image.png"
            cv2.imwrite(output_image_path, canvas)

            return render_template('result.html', image_path=output_image_path, text=text)

if __name__ == '__main__':
    app.run(debug=True)
