# GSS-ML-Model-Suite
GSS MLOps ML Model Suite is a central repository of re-usable, pre-trained ml models. The code base can be easily used to experiment around different bussines use cases with minimal effort of developing & training the model. This repository can be used for different computer vision (image data) and nlp (text data) use cases which are listed below:

#### Below are the berief description of every individual module which can help the user to understand which module to use to solve their business problem statement.

## Computer Vision Use Cases [Image Dataset]
#### 1. Image Enhancement
Image enhancement module can be used for do the following moddification in a given image: 
1. Resize the image
2. Add the borders 
#### 2. Image Similarity
The image similarity module calculates the similarity between given two images. It gives the similarity score between two images. It is based on CNN (convolutional Neural Networks) to extract the feature and then cosine similarity to calculate the similarity between the images.

#### 3. Object Detection
Object Detection module is used to detect any object in a given image. An object detection model identifies the object (what the object is ? Is it a dog or a cat?) and also locates the object in the image. It will give a bounding box around the object of interest. <Add an example image>
## NLP Use Cases [Text Dataset]
#### 1. Text Summarisation
Text summarisation module outputs the summary of the input paragraph. This module can be used to generate a brief summary of long text phrases such as product discrption, long feedback & comments. Such summarisation module makes it easier for the end user to easily understand the overall context with the help of the sumamrised output. 
#### 2. Language Detection
 Language detection module detects the major language of your text data. It uses python library to detect the language. This module can be helpful if user wants to filter out the other language or wants to segregate the data based on their language.
#### 3. Text Similarity
Text similarity calcualtes the similarity score between two given phrases (text). This module uses bert based nlp model generate the vector and then uses cosine similarity to calculate the similarity scores. This module helps the user to identify similar products based on their description, recommened relevant items/products etc. 
#### 4. Sentiment Analysis
Sentiment Analysis module analyses the sentiment of the given phrase. The phrase can be a review, a feedback, a comment etc, this module will analyse the overall sentiment of the phrase and tell you whether the sentiment is positive, negitive or neutral. 
#### 5. Text Classification
Text classification model classifies the a given text data (sentence/phrase) into defined classes. WIP

# Installation
```
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
pip install requirement.txt
```

