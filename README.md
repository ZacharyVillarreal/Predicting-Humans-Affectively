![](images/title_image.png)

# Emotion-AI: Predicting Humans Affectively

**Using Audio and Images to Predict Human Emotion**<br>
Zachary Villarreal<br>
[LinkedIn](#https://www.linkedin.com/in/zachary-p-villarreal/) | [GitHub](#https://github.com/ZacharyVillarreal) | [zpvillarreal@gmail.com](#zpvillarreal@gmail.com)


<a name="Table-of-Contents"></a>
## Table of Contents
* [Background](#background)
    * [Motivation](#Motivation)
    * [Project Goal](#Project-Goal)
* [Data](#Data)
    * [Data Allocation](#Data-Allocation)
    * [Pipeline](#Pipeline)
* [Exploratory Data Analysis](#Exploratory-Data-Analysis)
    * [Audio Data](#Audio-Data)
    * [Image Data](#Image-Data)
* [Models](#Models)
    * [Convolutional Neural Networks](#Convolutional-Neural-Networks)
    * [Audio Model](#Audio-Model)
    * [Image Model](#Image-Model)
* [Dash App](#Dash-App)
* [Conclusion](#Conclusion)


<a name="background"></a>
## Background
<a name="Motivation"></a>
### Motivation

Coming from a background in Biology and research focusing on human disease, I have always been interested in the interaction between humans and data science, specifically how our psychology influences data patterns. Emotions hold a dominant influence not only on how we live and interact with others but also on the actions we take and the choices we make. For example, feelings of happiness help us make decisions and allows us to consider a larger set of options, to be more likely to purchase goods or visit stores. While feelings of sadness, or fear, may stop an individual from stepping outside of their comfort zone and prevent them from taking action.

I often wondered, how can we be able to use machines to detect human affect, human emotion? This is when I started researching what is known as *Emotion AI*. Emotion AI, is a subset of artifical intelligence that aims to be able to predict human behavior, human emotion, in the same way that we, as humans, do. I knew I wanted to be able to build on already existing Emotion AI practices to be able to predict human emotion more effectively. 
<a name="Project-Goal"></a>
### Project Goal

This project takes in both facial images and audio samples in order to create a more accurate representation of human emotion through predictive modeling. Providing another dimension to the already existing research around Emotion AI that exists today. 

<a href="#Table-of-Contents">Back to top</a>


<a name="Data"></a>
## Data
<a name="Data-Allocation"></a>
### Data Allocation
* Audio:
    * [Crema-D](#https://www.kaggle.com/ejlok1/cremad): Crowd Sources Emotional Multimodal Actors Dataset
    * [RAVDESS](#https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio): Ryerson Audio-Visual Database of Emotional Speech and Song
    * [SAVEE](#https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee): Surrey Audio-Visual Expressed Emotion
    * [TESS](#https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess): Toronto Emotional Speech Set

> Note: Four sets of data were utilized for emotion recognition via audio files, for reason that each data set consisted only of a couple thousand audio files. When accumulated together, these databases allowed for a more accurate predictive model. In addition to, an end goal of the models we are building are for live use within an app. Four different audio sources can provide different levels of pitch, sound, volume, etc. which can help combat variances within users' microphones.

* Images:
    * [FER2013](#https://www.kaggle.com/deadskull7/fer2013): Facial Expression Recognition Competition

<a href="#Table-of-Contents">Back to top</a>
<a name="Pipeline"></a>
### Pipeline

##### Audio:
* Load and Clean Data:
    * Have all locations of audio files within one dataframe
* Exporatory Data Analysis
* Feature Extration:
    * [MFCC](#https://en.wikipedia.org/wiki/Mel-frequency_cepstrum#:~:text=Mel%2Dfrequency%20cepstral%20coefficients%20(MFCCs,%2Da%2Dspectrum%22)): Mel-frequency cepstrum: distinct units of sound
* Machine Learning Models:
    * Convolutional Neural Networks
* Convolutional Neural Network:
    * Predict Male / Female
    * Predict probability of emotion
* Integrate into Dash app<br>

![](images/audio_pipeline.png)

<a href="#Table-of-Contents">Back to top</a>

##### Images:
* Load and Clean Data:
    * Have all locations of image files within one dataframe
* Exporatory Data Analysis
* Feature Extration:
    * Pixel intensities of grayscaled images
* Machine Learning Models:
    * Convolutional Neural Networks
* Convolutional Neural Network:
    * Predict probability of emotion
* Integrate into Dash app
    * Use OpenCV for facial recognition
    * Use OpenCV for Male / Female Prediction
    
![](images/image_pipeline.png)

<a href="#Table-of-Contents">Back to top</a>

<a name="Exploratory-Data-Analysis"></a>
## Exploratory Data Analysis

Let's take a quick look at all the data gathered for this project.<br>
<a name="Audio-Data"></a>
**Audio Data:**<br>
* 13,000 Audio Clips
* 7 Emotions: Angry, Sad, Happy, Surprised, Fear, Neutral, Disgust
* 2 Sexes: Male, Female

Distribution of Emotions             |  Distribution of Sex
:-------------------------:|:-------------------------:
![](images/distribution_of_emotions_audio.png)  |  ![](images/distribution_of_sexes_audio.png)

That's pretty interesting, I am now interested in looking at how things like pitch or volume is affected by the sex of the speaker. We can look at it from two perspectives. One being the wave plot, which depicts the amplitude of the sound over time, and the MFCC, or the *Mel Frequency Cepstral Coefficient*, which are the features I am feeding into my neural network. MFCC scales the sound frequencies to make the features better represent what we humans hear.

Waveplot             |  MFCC (Male) | MFCC (Female)
:-------------------------:|:-------------------------: | :-----------------------------:
![](images/male_vs_female_angry_audio.png)  |  ![](images/male_mfcc_audio.png) | ![](images/female_mfcc_audio.png)

While although there are subtle differences in the MFCC plots, it might be hard to discern the key differences between both the male MFCC and the female MFCC of their respective *angry* audio clips. It is much more apparent in the waveplot, that there is a difference in the levels of sound provided by the two voice actors. The male's voice is distinctly louder than the female's. 

<a href="#Table-of-Contents">Back to top</a><br>
<a name="Image-Data"></a>

**Image Data:**<br>
* 36,000 Images
* 7 Emotions

![](images/distribution_of_emotions_images.png)

There seems to be a much less even distribution of emotions over the image data. This could possibly cause a problem when it comes to training our neural networks later in this project, but we will deal with that later. Next, let's take a look at how the original images come in.

Examples of Emotional Expressions:<br>

Angry |  Happy | Sad | Disgust | Surprise | Neutral | Fear|
:-------------------------:|:-------------------------: | :-----------------------------: | :---: | :---: | :--: | :--:
![](images/angry_example.png) | ![](images/happy_example.png) |  ![](images/sad_example.png) | ![](images/disgust_example.png) | ![](images/surprise_example.png) | ![](images/neutral_example.png) | ![](images/fear_example.png)

Interesting, it seems that the images are greyscaled, we can use this to teach our convolutional neural networks.

<a href="#Table-of-Contents">Back to top</a><br>
<a name="Models"></a>
## Models
---
![](images/CNN_example.png)
<a name="Convolutional-Neural-Networks"></a>
### Convolutional Neural Networks

At a high level, a Convolutional Neural Network (CNN) is a deep learning algorithm that takes in an image as an input and is able to perform classification. However, unlike a regulare artifical neural network, a CNN has pre-processing steps that apply convolutions that highlight features of the inputted image to help distinguish between the various classes we are trying to learn. Because we are using the MFCC of the audio files, we can treat these as images as it provides more information and allows the use of CNNs. We will also be using a distinct CNN for emotion classification via facial images. 

<a href="#Table-of-Contents">Back to top</a><br>
<a name="Audio-Model"></a>
### Audio Model

Audio CNN:
1. Image was fed through 4 distinct Convolution layers with MaxPooling and BatchNormalization
1. The image data was then flattened.
1. The flattened data was fed into 3 dense layers and a dropout layer to combat overfitting.
1. The final layer returned a prediction for one of 7 emotions over 2 sexes, so 14 distinct classes.

<p align='left'>
<img src="images/audio_cnn_summary.png" width="700" height="600"> 
</p>


Let's take a look to see how our Audio CNN did overall.

Audio CNN Accuracy  |  Audio CNN Confusion Matrix
:-------------------------:|:-------------------------:
![](images/cnn_model_accuracy.png)  |  ![](images/audio_cnn_confusion_matrix.png)

I was able to achieve a 67% accuracy on my validation data. While at first this might not seem very high, we must remember that we are trying to predict between 14 distinct classes, which when randomly chosen, has approximate a 7% of being right. Thus, we were able to achieve a 60% increase in the model's accuracy.

<a href="#Table-of-Contents">Back to top</a><br>
<a name="Image-Model"></a>
### Image Model

Image CNN:
1. Image was fed through 8, 4 distinct 2-dimentional, Convolution layers with MaxPooling and BatchNormalization
1. The image data was then flattened.
1. The flattened data was fed into 3 dense layers and a dropout layer to combat overfitting.
1. The final layer returned a prediction for one of 7 emotion labels.


![](images/image_cnn_summary.png)

Let's take a look to see how our Image CNN did overall.

Image CNN Accuracy  |  Image CNN Confusion Matrix
:-------------------------:|:-------------------------:
![](images/image_cnn_accuracy.png)  |  ![](images/image_cnn_cf.png)

I was able to achieve a 62% accuracy on my validation data. However, unlike the audio data, the image data was not pre-labeled for the sex of the individual being photographed, so we are only trying to predict between 7 distinct emotions. Thus, we have a 48% increase in accuracy from our baseline of 14%.

<a href="#Table-of-Contents">Back to top</a><br>
<a name="Dash-App"></a>
## Dash App


I then thought to myself, "how can we use these neural networks that I built in a live setting?" I then created a web application using Dash and Flask, where users can input either an image file or an audio file. Once the app recognizes the input, it will run it through its respective Convolutional Neural Network and will output the predictions, both sex and emotion, for both images and audio. In addition, it will output the opposite type of file, image to audio and vice versa, depicting a representation of that predicted emotion. 

Live images needed further investigation in order to fit the input parameters of the image CNN. Thus, OpenCV's [haarcascade-frontalface-default](#haarcascade_frontalface_default.xml) aws used in order to extract the facial from the live input. Once the face image was recognized, the image was contorted and greyscaled in order to meet the CNN parameters. In addition, [Py-Agender](#https://pypi.org/project/py-agender/) was used in order to predict sex of the individual in the photo. 

Here is an example of when we input an image file.

![](images/female_angry_app.png)
This app could be used for things, as simple as, trying to recognize emotions on images of sound bytes of yourself, to being able to teach individuals the importance of facial language or how the inflection in our voice is depicted by other people.

<a href="#Table-of-Contents">Back to top</a><br>
<a name="Conclusion"></a>
## Conclusion

In conclusion, the methodologies implemented in this project allowed for an increase in prediction accuracy by 40-60% depending on the type of input being tested. The justification for the limitation in accuracy could stem from two major factors: 1) the initial training dataset was relatively small; 2) either or both of the datasets were biased in terms of user-labeled data. Although the accuracy was not extremely high, the current models do produce mostly correct predictions when it comes to live data. 
