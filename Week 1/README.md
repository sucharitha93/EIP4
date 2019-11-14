**EIP 4  Week 1  Assignment**

In [0]:
score = model.evaluate(X_test, Y_test, verbose=0)
In [64]:
print(score)
[0.031204893102007917, 0.9909]

**Definitions**

>**Convolution**

Convolution of an image can be defined as the process of multiplying the image with an operator (which is called a kernel) to produce a desired effect of enhancement or feature extraction.

>**Filters/Kernels**

Filter or Kernel is the matrix of values which is used to perform convolution by modifying each pixel value by considering the influence of the neighboring pixels.

>**Epochs**

Epoch is the measure used to describe the step when the dataset completes one cycle of training in the algorithm. In other words, it is defined as the completion of processing the batch of data one time.

>**1x1 Convolution**

The 1x1 Convolution operation involves combining all the values in each channel to produce a single value for each pixel in the image. This type of operation can also be called channel wise pooling.

>**3x3 Convolution**

The 3x3 convolution operator is a two dimensional array of values which is called the kernel. The most widely adopted standard kernel size is 3x3. 

>**Feature Maps**

Feature Map is the output containing combined features which are obtained from a convolution layer. 

>**Activation Function**

Activation function converts the squashed linear input into non linear values.

>**Receptive Field**

The representative area of the feature in the image which contains the feature is called Receptive Field. Global and Local receptive fields are two types.
