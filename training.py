#!/usr/bin/env python
# coding: utf-8

# In[25]:


import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from shutil import copyfile
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adamax


# In[27]:


import os
# Set the path to your dataset directory
dataset_path = r"C:\Users\ASUS\LUNGS CANCER\lung_colon_image_set"
# Walk through the directory and print the directory names
for dirName, _, fileNames in os.walk(dataset_path):
    print(dirName)


# In[29]:


import os

# Set the path to your dataset directory
dataset_path = r"C:\Users\ASUS\LUNGS CANCER\lung_colon_image_set"
# Walk through the directory and print the directory names
for dirName, _, fileNames in os.walk(dataset_path):
    count = 0
    print("Directory:", dirName)
    
    # Optionally, you can count the number of files in the directory
    count = len(fileNames)
    print("Number of files in this directory:", count)


# In[47]:


def class_countPlot(datasets):
    # Check if 'datasets' is not a list and convert it to a list if needed
    if not isinstance(datasets, list):
        datasets = [datasets]  # Wrap it in a list
        
    class_counts = {}
    
    if datasets:
        for dataset in datasets:
            for folder in os.listdir(dataset):
                folder_path = os.path.join(dataset, folder)
                if os.path.isdir(folder_path):
                    files = os.listdir(folder_path)
                    images_count = len(files)
                    class_counts[folder] = images_count
                    
                else:
                    print(f"Folder {folder_path} does not exist.")

        # plotting the counts of each class
        class_names = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Get a colormap  
        cmap = plt.get_cmap("tab10")  # Choose a colormap (you can change 'viridis' to any other)  
        # Create a color for each count based on the colormap  
        colors = [cmap(i / len(counts)) for i in range(len(counts))]
        
        plt.figure(figsize=(10, 6))  
        plt.bar(class_names, counts, color= colors)  
        plt.xlabel('Classes')  
        plt.ylabel('Number of Images')  
        plt.title('Count of Images in Each Class')  
        plt.xticks(rotation=45)  
        plt.grid(axis='y')  
        
        # Show the plot  
        plt.tight_layout()  
        plt.show() 
        
    else:
        print("Warning: dataset(s) empty!!")


# In[49]:


def split_data(source, training, testing, split_size):
    # Ensure the destination directories exist
    os.makedirs(training, exist_ok=True)
    os.makedirs(testing, exist_ok=True)

    for folder in os.listdir(source):
        files = []
        folder_path = os.path.join(source, folder)
        
        # Create subdirectories for training and testing classes
        os.makedirs(os.path.join(training, folder), exist_ok=True)
        os.makedirs(os.path.join(testing, folder), exist_ok=True)

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.getsize(file_path) > 0:
                files.append(file)
            else:
                print(f"{file} has not enough pixels to represent it as an image, seems corrupted so ignoring.")

        training_length = int(len(files) * split_size)
        testing_length = len(files) - training_length

        # Shuffle and split files
        training_set = shuffle(files[:training_length])
        testing_set = shuffle(files[training_length:])

        # Copy files to the training set
        for file in training_set:
            file_path = os.path.join(folder_path, file)
            destination_path = os.path.join(training, folder, file)
            copyfile(file_path, destination_path)

        # Copy files to the testing set
        for file in testing_set:
            file_path = os.path.join(folder_path, file)
            destination_path = os.path.join(testing, folder, file)
            copyfile(file_path, destination_path)


# In[51]:


def plot_accuracy_loss(history):
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'], label = "train")
    plt.plot(history.history['val_accuracy'], label = "validataion")
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend(loc = 'upper left')

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'], label = "loss")
    plt.plot(history.history['val_loss'], label = "val_loss")
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(loc = 'upper left')
    plt.show()


# In[53]:


optimizer = Adamax(learning_rate=0.001)
epochs = 100
img_size = (200, 200)
batch_size = 256
split_size = 0.8

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    restore_best_weights=True
)

# ModelCheckpoint callback to save the best model weights  
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "DenseNet_noTop_MRI.keras",  # Path to save the model weights
    monitor='val_accuracy',       # Which metric to monitor  
    save_best_only=True,         # Save only the best model  
    mode='max',                  # We want to maximize the monitored metric  
    verbose=1                    # Print a message when the model
)

# Source directories (update these paths according to your local setup)
COLON_SOURCE_DIR = r"C:\Users\ASUS\LUNGS CANCER\lung_colon_image_set\colon_image_sets"
LUNG_SOURCE_DIR = r"C:\Users\ASUS\LUNGS CANCER\lung_colon_image_set\lung_image_sets"
TRAINING_DIR = r"C:\Users\ASUS\LUNGS CANCER\lung_colon_image_set\training"
TESTING_DIR = r"C:\Users\ASUS\LUNGS CANCER\lung_colon_image_set\testing"
DESTINATION = r"C:\Users\ASUS\LUNGS CANCER"  # Destination for any outputs


# In[55]:


datasets = [COLON_SOURCE_DIR, LUNG_SOURCE_DIR]


# In[57]:


class_countPlot(datasets)


# In[59]:


split_data(COLON_SOURCE_DIR, TRAINING_DIR, TESTING_DIR, split_size)
split_data(LUNG_SOURCE_DIR, TRAINING_DIR, TESTING_DIR, split_size)


# In[61]:


class_countPlot(TRAINING_DIR), class_countPlot(TESTING_DIR)


# In[64]:


# Make datagen for Train generator
train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Train generator
train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                    target_size = img_size,
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    class_mode = "categorical",
                                                    color_mode = "rgb",
                                                    subset = "training"
                                                   )

# validation generator
validation_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                         target_size = img_size,
                                                         batch_size = batch_size,
                                                         shuffle = False,
                                                         class_mode = "categorical",
                                                         color_mode = "rgb",
                                                         subset = "validation"
                                                        )
# Test generator
test_generator = test_datagen.flow_from_directory(TESTING_DIR,
                                                  target_size = img_size,
                                                  batch_size = batch_size,
                                                  shuffle = False,
                                                  class_mode = "categorical",
                                                  color_mode = "rgb",
                                                 )


# In[66]:


class_indices = train_generator.class_indices
class_names = list(class_indices.keys())


# In[68]:


fig, axs = plt.subplots(1, 5, figsize = (20, 4))
axs = axs.flatten()
train_batch = next(train_generator)

for i ,ax in enumerate(axs):
    ax.imshow(train_batch[0][i])
    label = tf.argmax(train_batch[1][i])
    ax.set_title(class_names[label])
plt.show()


# In[70]:


base_model = tf.keras.applications.DenseNet121(weights = "imagenet", include_top = False, pooling = "avg", input_shape = (img_size[0], img_size[1], 3))


# In[72]:


base_model.trainable = False


# In[74]:


model = base_model.output
model = tf.keras.layers.BatchNormalization()(model)
model = tf.keras.layers.Dropout(0.5)(model)
model = tf.keras.layers.Dense(5, activation = "softmax")(model)
model = tf.keras.models.Model(inputs = base_model.input, outputs = model)


# In[76]:


model.summary()


# In[78]:


model.compile(optimizer, loss= 'categorical_crossentropy', metrics= ['accuracy'])


# In[ ]:


history = model.fit(x= train_generator,
                    epochs= epochs,
                    verbose= 1,
                    validation_data= validation_generator,
                    callbacks= [early_stopping, model_checkpoint]
                    )


# In[ ]:


plot_accuracy_loss(history)


# In[84]:


model.load_weights("DenseNet_noTop_MRI.keras")


# In[86]:


model.evaluate(test_generator)[1]


# In[88]:


# Make predictions  
predictions = model.predict(test_generator)  

# If you want to extract class indices based on expectations  
predicted_classes = tf.argmax(predictions, axis=-1)  # Get predicted class indices


# In[90]:


true_classes = test_generator.classes

print("Classification Report:")  
print(classification_report(true_classes, predicted_classes)) 

cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_names)
disp.plot()


# In[ ]:




