# GSS-ML-Model-Suite
GSS MLOps ML Model Suite is a central repository of re-usable, pre-trained ml models. The code base can be easily used to experiment around different bussines use cases with minimal effort of developing & training the model. This repository can be used for different computer vision (image data) and nlp (text data) use cases which are listed below:

#### Below are the berief description of every individual module which can help the user to understand which module to use to solve their business problem statement.

### Computer Vision (Image & Video Data)
Use cases that relate to image and video data types fall under the category of Computer Vision in the ML world.
1. Image Enhancement: This module can be used to perform various modifications for a given image:
a. Resize the image to maintain a user defined aspect ratio, height, or width
b. Center & zoom the image based on user defined percentage/ratio
c. Add user defined color borders to images to maintain aesthetic quality
2. Image Similarity: This module calculates the similarity between two input images and provides a score [0 to 1] based on latent features and embeddings.
3. Object Detection [Image]: This module can be leveraged to detect ~100 different object types within a given image. It not only provides the type of object but also a confidence score and draws a bounding box surrounding the objects identified.
4. Semantic Segmentation [Image]: This module can be leveraged to segment an image into various classes. The objective is to label each pixel of an input image with a corresponding class of what is being represented.
5. Image Classification: This module provides the user with a modularized code base to develop their own classification model given a set of images into user-defined classes.
6. Object Detection & Tracking [Video]: This module can identify ~100 different objects, draw bounding boxes and track them across keyframes in a given video. This is specially useful for traffic monitoring, robotics, self-driving vehicles, etc.
7. Text to Image Generation: This module leverages Stable Diffusion to generate images based on input text provided by the user. This can be helpful for data augmentation, as well as creating reference images in labeling SOPs for agents to have visual guidance.
### Natural Language Processing (Text Data)
Use cases that relate to text data type fall under the category of Natural Language Processing in the ML world.
1. Text Summarization: This module generates a summarized output of a given text input and can be utilized to obtain a brief summary of long text phrases such as product descriptions, long feedback & comments.
2. Language Detection: This module detects the distribution of text data across languages by passing the input through an implementation of Google’s language detector. This can be extremely useful to segregate text data across languages as a pre-processing step.
3. Language Translation: This module enables translation of text input across various source<>target pairs by leveraging open-source Machine Translation algorithms developed by Meta, Google, etc.
4. Named Entity Recognition: This module identifies specific keywords in a given phrase or paragraph. In an input text, this pre-trained model can identify the Name of the person, Name of the place (city, country, state), and dates. This can be used to extract such key-value pairs from data for further usage. 
5. Multi-lingual Text Similarity: This module calculates the similarity score between two given input texts. This can be utilized to identify similar products based on their description, de-duplicate bugs, etc.
6. Sentiment Analysis: This module analyzes the sentiment of the given text input and provides various forms of categorization [positive, negative, anger, sorrow, happiness, ratings on a scale of 1 to 5] that can be really useful to understand latent themes and identify specific focus areas for support channels.
7. Multi-class Text Classification: This module provides the user with a modularized code base to develop their own classification model given a set of text data into user-defined classes. This can be useful for product categorization, classifying tickets into different issue types and priority buckets, etc.
8. Zero-shot Text Classification: This allows users to  to associate an appropriate label with a piece of text without any model training required. This is particularly useful for annotation of text data based on user provided themes.
