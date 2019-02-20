# Multi-label Classification Using a Variation of VGGNet

## Introduction
Multi-label classification is a classification problem where multiple target labels can be assigned to each observation instead of only one in multi-class classification. It can be regarded as a special case of multivariate classification or multi-target prediction problems, for which the scale of each response variable can be of any kind, for example nominal, ordinal or interval.
<p>Two different approaches exist for multilabel classification. On the one hand, there are algorithm adaptation methods that try to adapt multiclass algorithms, so they can be applied directly to the problem. On the other hand, there are problem transformation methods, which try to transform the multilabel classification into binary or multiclass classification problems.</p>
<p>In this project we use the Binary Relevance method. The binary relevance method (BR) is the simplest problem transformation method. BR learns a binary classifier for each label. Each classifier C1, . . ., Cm is responsible for predicting the relevance of their corresponding label by a 0/1 prediction: Ck: X → {0, 1}, k = 1, . . ., m</p>
<p>These binary predictions are then combined to a multilabel target.
The VGG network architecture was introduced by Simonyan and Zisserman in their 2014 paper, Very Deep Convolutional Networks for Large Scale Image Recognition. They achieved state-of-the-art results in the ILSVRC-2014 challenge securing first and second in localization and classification tracks respectively.
This network is characterized by its simplicity, using only 3×3 receptive fields (which is the smallest size to capture the notion of left/right, up/down, center) stacked on top of each other in increasing depth. Reducing volume size is handled by max pooling. Two fully-connected layers, each with 4,096 nodes are then followed by a softmax classifier.</p>
<p>The architecture of our neural network is heavily inspired by that of VGGNet. We use the same concept of small 3x3 receptive fields with depth increasing after each layer, but the only difference is that of the depth of the network.</p>

## Examples
![Sample 1](https://github.com/csaiprashant/multilabel_classification_vggnet/blob/master/examples/01.png)
![Sample 2](https://github.com/csaiprashant/multilabel_classification_vggnet/blob/master/examples/10.png)

## Files in the Repository:
- /examples/ - Contains a few examples output of our SmallerVGGNet on our test images.
- __init__.py - Blank python script to initialize Python packages.
- Accuracy and Loss plot.png - Plot showing the accuracy and loss during the neural network training.
- fashion.model - The multi-label classifier.
- mlb.pickle - Pickle file containing the labels of our training set.
- projectreport.pdf - Project Report.
- README.md
- smallervggnet.py - Architecture of our neural network.
- test_network.py - Python script for applying the trained classifier on a test image. 
- train_network.py - Python script for training the classifer.

### For full report, please refer ![https://github.com/csaiprashant/multilabel_classification_vggnet/blob/master/projectreport.pdf]
