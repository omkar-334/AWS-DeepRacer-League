## 3.1 Case Studies that use ML

Supervised learning

* Using machine learning to predict housing prices in a neighborhood, based on lot size and the number of bedrooms.

Unsupervised learning

* Using machine learning to isolate micro-genres of books by analyzing the wording on the back cover description.

Deep neural network

* While this type of task is beyond the scope of this lesson, we wanted to show you the power and versatility of modern machine learning. You will see how it can be used to analyze raw images from lab video footage from security cameras, trying to detect chemical spills.

House Price Prediction

* **Regression:** A common task in supervised machine learning used to understand the relationship between multiple variables from a dataset.
* **Continuous:** Floating-point values with an infinite range of possible values. This is the opposite of categorical or discrete values, which take on a limited number of possible values.
* **Hyperplane:** A mathematical term for a surface that contains more than two planes
* **Plane:** A mathematical term for a flat surface (like a piece of paper) on which two points can be joined by drawing a straight line.

## 3.2 Using ML to predict a book's genre

Continuous values - regression (SL)

unlabeled values - kmeans (USL)

D**ata cleaning and exploration**

For this project, you believe capitalization and verb tense will not matter, and therefore you remove capitals and convert all verbs to the same tense using a Python library built for processing human language. You also remove punctuation and words you don’t think have useful meaning, like *'a'* and ‘the' . The machine learning community refers to these words as *stop words.*

**Data preprocessing**

Before you can train the model, you need to do a type of data preprocessing called  *data vectorization* , which is used to convert text into numbers.

You transform this book description text into what is called a *bag of words representation* , so that it is understandable by machine learning models.

* **Bag of words** : A technique used to extract features from text. It counts how many times a word appears in a document (corpus), and then transforms that information into a dataset.
* **Data vectorization** : A process that converts non-numeric data into a numerical format so that it can be used by a machine learning model.
* **Silhouette coefficients** : A score from -1 to 1 describing the clusters found during modeling. A score near zero indicates overlapping clusters, and scores less than zero indicate data points assigned to incorrect clusters. A score approaching 1 indicates successful identification of discrete non-overlapping clusters.
* **Stop words** : A list of words removed by natural language processing tools when building your dataset. There is no single universal list of stop words used by all-natural language processing tools.

## 3.3 Using ML to detect spills

**Data vectorization (converting to numbers)**

* Many models require numerical data, so you must transform all of your image data needs to be transformed into a numerical format. Python tools can help you do this automatically.
* In the following image, you can see how each pixel in the image immediately below can be represented in the image beneath it using a number between 0 and 1, with 0 being completely black and 1 being completely white.

**CNN (convolutional neural network)**

Neural networks are beyond the scope of this lesson, but you can think of them as a collection of very simple models connected together. These simple models are called  *neurons* , and the connections between these models are trainable model parameters called  *weights* .

Convolutional neural networks are a special type of neural network that is particularly good at processing images.

list of common metrics:

* Accuracy
* Confusion matrix
* F1 score
* False positive rate
* False negative rate
* Log loss
* Negative predictive value
* Precession
* Recall
* ROC Curve
* Specificity
