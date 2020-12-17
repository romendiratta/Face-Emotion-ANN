##### UW-Madison: CS539 Fall 2020
# Facial Expression Classification with Images
#### By Rohan Mendiratta, Huanran Li 
#### <a href="http://github.com/romendiratta/Face-Emotion-ANN">View Repository</a>

<br/>

### **I. Introduction**
In the recent decade, convolutional neural networks have shown a tremendous advantage in picture classification tasks. The idea of a building convolutional neural network to extract key points in the face and determine emotion based on this has previously been proposed (Mehendale). The idea of using a pre-existing neural network and using transfer learning to classify human emotion has also been proposed as well (Gilligan). Both of these ideas and accompanying research have shown that this task can be accomplished in different ways with high accuracy. The problem with this though is that this source code is not publicly available for use. In this project, the aforementioned research will be used as inspiration to build a convolution neural network to solve this problem with a high degree of accuracy. 

With the advance of technology in today’s age, face-to-face contact is less common. For humans, it has always been easy to detect emotions, but the same cannot be said for machines. This is why the idea of building an algorithm to identify human emotion based on the image of someone’s face has been proposed. The utilization of this algorithm can be significant in many ways. With this algorithm, the world can take advantage to improve the human quality of life. This applies to many facets of life. Automated emotion recognition is significant in a few fields: Robotics - To design smart and collaborative service robots that can interact with humans appropriately; Marketing - To create specialized advertising based on the emotional state of a person; Education - To improve learning processes and knowledge transfer; Entertainment - To recommend appropriate entertainment for a target audience based on emotional state.

The idea of automated emotion recognition is not new. In a related work, attempting to identify human emotion through electrocardiography, galvanic skin response, heart rate variability, etc. has been proposed and successful to a high degree (Dzedzickis). In a similar work, emotion recognition has been attempted through the processing of text (Ezhilarasi). Within both of these works, the classification of emotion has been done through active data collection. On the other hand, imaging of face can be done passively, which makes the proposed algorithm very powerful as no interaction is required in order to detect emotion.

### **II. Methods**

#### **Data**
The data used is from New York University created by Professor Rowe with his machine learning graduate students [?]. It consists of 13,690 cropped human faces with 8-class labels. The images are 3 channels and sized 256 by 256 pixels. Each face was labeled with anger, contempt, disgust, fear, happiness, neutral, sadness, or surprise.

<img src='./resources/data-distribution.png'>
<br></br>

For the preprocessing of the data, the images are resized to 128 by 128 pixels to reduce convergence time, while still maintaining a high degree of accuracy. Transformation techniques such as inverting and rotating will be used on the dataset while preprocessing to overcome the data imbalance (as shown in Figure 1) and avoid model overfitting. The data is split into 56% training, 14% validation, and 30% testing sets using the scikit-learn library. 

#### **Algorithm**
The convolutional neural net has been constructed using transfer learning. ResNet50 and InceptionV3 were both used separately as base models and their performances are individually determined. Both models have been initialized with ImageNet weights and additional layers have been added. The base layers are not trained, while the additional layers are trained specifically for emotion classification. Both models were compiled using the categorical cross entropy loss function, as this function is optimized for multi-category classification. The detailed structure is shown at Figure 2 (left).

To speed up the training process, we developed the second model as shown in Figure 2 (right) which feeds the data though the pre-trained network once and uses the extracted feature as input to train the last few layers. Because of its nature, this network cannot take any augmented data because the feature was extracted only once before training.

<img src='./resources/models.png'>
<br></br>
These models have been built on the Python platform while utilizing the TensorFlow and Keras libraries to build the models. Image pre-processing functions and model constructing is done using Keras functions. The following categories are used to evaluate model performance: Training Accuracy; Validation Accuracy; Test Accuracy; Training Time. Training accuracy and validation accuracy will be indicators of whether the model has completed training. Test accuracy is the main indicator of how the model will perform on unseen images and is the main factor in determining which model is better suited for this classification task. Training time is a small factor here. As long as training time is not unfeasible, it can be disregarded as the model only needs to be trained once. Training time is relative to the Google Colab platform, which both models are trained on. GPU processing is enabled and the NVIDIA TESLA K80 that is offered on Google Colab is used while training both models to improve convergence time. 