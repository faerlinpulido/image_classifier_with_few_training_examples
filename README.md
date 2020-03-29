# Image Classifier With Few Training Examples

In this notebook, we build an image classifier using only very few training examples (~1000 images per class). In particular, we define our model by attaching a new classifier (randomly initialized) on the pre-trained convolutional base of the InceptionV3 network. This model is then trained on images with real-time augmentations (e.g. rotation, shear, zoom, width and height shifts, horizontal flip). In addition, we make prediction using this model with a technique called test-time augmentation (TTA); that is, given an input image, our model makes predictions on multiple augmented copies of that image and return an ensemble prediction via soft voting. We train our model on the Food-101 dataset (https://kuanghuei.github.io/Food-101N/). The Food-101 dataset is designed for learning to address label noise with minimum human supervision (Lee, 2018), but we use it here because it is a dataset with very few instances for each class (~1000 images per class). It is made up of 101 food categories with 101000 images where each image has been rescaled to have a maximum side length of 512 pixels.

We define our model as follows: we take the pretrained InceptionV3 or VGG16 network, remove its trained classifier, and attach a fully connected classifier with an output layer appropriate for our purposes. That is, our model is made up two parts: a convolutional base and a classifier.

We begin by investigating how to effectively train our model. For the sake of lower computational time, we perform various ways of training our model using only images from 10 classes. We explore the following ways of training our model:

- Freeze the entire convolutional base then re-train (call this the baseline model)
- Starting from the trained baseline model, unfreeze a few of the top layers of the convolutional base then re-train
- Starting from the trained baseline model, unfreeze the entire convolutional base then re-train

We evaluate each of these ways of training on the test set. We selet the one with the highest accuracy and use it to train a model using all 101 classes. Between InceptionV3 and VGG16, we find that using the convolutional base of InceptionV3, unfreezing it entirely and re-training it yields a model with the highest accuracy. With this approach trained on the entire dataset, our model achieves 80.8% accuracy on the test set. With test-time augmentation, this accuracy increases to 83.2%.

[Click here](https://github.com/faerlinpulido/image_classifier_with_few_training_examples/blob/master/notebook.ipynb) to go directly to the project's python notebook. 

By Faerlin Pulido
