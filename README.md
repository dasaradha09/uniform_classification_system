# **College uniform classification system**


 
## **Description**

College uniform classification system is a deep learning-based image classification system that identifies whether a person in an image is wearing a uniform or not. The project leverages Convolutional Neural Networks (CNNs) and includes data augmentation to improve accuracy, even with a small dataset.

----

## **Demeo Video**

----------------

## **Key Features**

1. **Binary Image Classification**:  
   - Classifies images into two categories:  
     - **Person Wearing Uniform**  
     - **Person Not Wearing Uniform**  

2. **Custom Dataset Creation**:  
   - Collected data using Python web scraping.  
   - Prepared a custom dataset specifically for this task.

3. **Data Augmentation**:  
   - Enhances model performance by generating diverse training samples through techniques like flipping, rotation, zooming, etc.

4. **Transfer Learning**:  
   - Leveraged the **VGG16** pre-trained model to improve accuracy and efficiency.  
   - VGG16 is particularly effective at capturing colors and textures in images, which is crucial for this project.

5. **User-Friendly Interface**:  
   - Provides a clear and simple interface for easy interaction.

6. **Flexible Input Options**:  
   - Users can classify images by:  
     - Uploading images from their computer.  
     - Using a live feed from their web camera.


-----

## **Model Architecture**

The model for this project is based on the **VGG16** architecture with transfer learning. The following steps describe the architecture:

1. **Base Model (VGG16)**  
   - The model uses the pre-trained **VGG16** network, which is loaded with **ImageNet** weights.  
   - The top layers of the VGG16 model are excluded (`include_top=False`), as we are only interested in the feature extraction part of the network.  
   - The input shape is set to `(128, 128, 3)` to match the resized images.

2. **Global Average Pooling**  
   - After the feature extraction layer, **GlobalAveragePooling2D** is used to reduce the spatial dimensions of the output, converting it into a one-dimensional vector while preserving important features.

3. **Fully Connected Layers**  
   - A **Dense** layer with 128 units and **ReLU** activation is added to introduce non-linearity and learn complex patterns.  
   - A **Dropout** layer with a rate of 0.5 is added to reduce overfitting during training.

4. **Output Layer**  
   - The final layer is a **Dense** layer with a single unit and **Sigmoid** activation to output a binary classification result (0 for "Not Wearing Uniform", 1 for "Wearing Uniform").

5. **Compilation**  
   - The model is compiled with the **Adam** optimizer, **binary_crossentropy** loss function (as it's a binary classification task), and accuracy as the evaluation metric.

### Code for Model Architecture:

```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

```

-------------


## **Technologies Used**

1. **Python**  
   - Used for writing the core logic and scripts of the project.

2. **NumPy**  
   - Used for converting images into arrays for processing.

3. **TensorFlow**  
   - Utilized for:  
     - Loading the pre-trained **VGG16** model.  
     - Implementing data augmentation techniques.  
     - Training the CNN model.

4. **Streamlit**  
   - Used to create a simple and interactive user interface.

--------

## **Installation and usage**

-------

## **Feature Improvements**

------

## **contact**

For any inquiries or questions, please contact us at sana000dasaradha@gmail.com

