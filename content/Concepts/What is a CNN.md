---
title: "What is a CNN"
draft: false
tags:
  - 
---

![WhatIsAConvolution](https://www.youtube.com/watch?v=KuXjwB4LzSA)

> CNN are neural network that have layers called **convolutional layers**. They have some type of specialization for being able to ***pick out or detect patterns***. This pattern detection is what makes CNNs so useful for _image analysis_.

# Convolution

**Convolution** - operation that transforms an input into an output through a _**filter**_ and a **_sliding window mechanism_.**

![[ConvolutionAnimation.gif|center|300]]
_Convolution animation example: a convolutional filter, shaded on the bottom, is sliding across the input channel._

> • Blue (bottom) - Input channel.
> • Shaded (on top of blue) - `3x3` convolutional filter or ***kernel***.
> • Green (top) - Output channel.

For each position on the blue input channel, the `3 x 3` filter does a computation that maps the shaded part of the blue input channel to the corresponding shaded part of the green output channel.

<u><b>Operation</b></u> 
At each step of the convolution, the **sum of the element-wise dot product** is computed and stored.

$$
\begin{align}
I_{nput} = 
\begin{pmatrix} a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23}\\
a_{31} & a_{32} & a_{33}
\end{pmatrix}\\
F_{ilter} =
\begin{pmatrix} b_{11} & b_{12} & b_{13} \\
b_{21} & b_{22} & b_{23}\\
b_{31} & b_{32} & b_{33}
\end{pmatrix} \\
\end{align} \\
\;\;\;Output = a_{11}b_{11}+a_{12}b_{12}+...+a_{33}b_{33}
$$

After this filter has convolved the entire input, we'll be left with a _new representation of our input_, which is now stored in the output channel. This output channel is called a **feature map**.

**Feature map** - output channels created from the convolutions.

>The word _feature_ is used because the outputs represent particular features from the image, like _edges_ for example, and these mappings **emerge** as the network learns during the training process and become more complex as we move deeper into the network.
# Conv Layer

> When adding a convolutional layer to a model, we also have to specify _how many filters_ we want the layer to have.

The number of filters determines the **number of output channels.**

![[DepthFeatureMaps.png|center|400]]
>_For example, if we apply `10` filters of size `5x5x3` to an input of size `32x32x3`, we will obtain a `32x32x10` output, where each depth component (red slice in image) is a **feature map**._

**Filters** - allow the network to detect _**patterns**_, such as _edges, shapes, textures, curves, objects, colors._

The _**deeper**_ the network goes, the **more sophisticated** the filters become. In later layers, rather than _edges and simple shapes_, our filters may be able to detect _**specific objects**_ like _eyes, ears, hair or fur, feathers, scales, and beaks._

In even _**deeper layers**_, the filters are able to detect **even more sophisticated objects** like _full dogs, cats, lizards, and birds._

<u><b>Hyperparameters</b></u> 

1. **Padding** - add values _"around"_ the image.
	-  helps _**preserve the input's spatial size**_  _(output size same as input)_, which allows an architecture designer to build _**deeper, higher performing**_ networks.
	- can help **retain information** by conserving data at the borders of activation maps.
2. **Kernel size** - dimensions of the sliding window over the input.
	- **massive impact** on the image classification task.
	- **small** kernel size
		- able to extract a **much larger amount of information** containing highly _**local features**_ from the input.
		- also leads to a _**smaller reduction in layer dimensions**_, which allows for a _deeper architecture_.
		- generally lead to **better performance** because able to **stack more** and more layers together to learn more and more complex features.
	- **large** kernel size
		- extracts _**less information**.
		- leads to a _**faster reduction**_ in layer dimensions, often leading to **worse performance**.
		- better suited to extract **larger** features.
3. **Stride** - how many pixels the kernel should be shifted over at a time.
	- ↗️ stride ↘️ size of output
	- **similar** impact than kernel size
		- ↘️ stride ↗️ size of output + **more features** are learned because more data is extracted.
		- ↗️ stride ↘️ size of output + **less feature** extraction.

>*Most often, a kernel will have odd-numbered dimensions -- `like kernel_size=(3, 3)` or `(5, 5)` -- so that a single pixel sits at the center, but this is not a requirement.*

>[!tip]- Conv2d with multiple input channels
> For two inputs, you can create two kernels. **Each kernel performs a convolution on its associated input channel.** The resulting output is added together as shown:  
> ![[Conv2dMultipleInputChannels.png|center|600]]
> 
> When using multiple inputs and outputs, a kernel is created for each input, and the process is repeated for each output. The process is summarized in the following image. 
> 
> There are two input channels and 3 output channels. For each channel, the input in red and purple is convolved with an individual kernel that is colored differently. As a result, there are three outputs. 
> ![[Conv2dMultipleInputAndOutput.png|center|600]]

> [!abstract] Parameters
> $$Parameters = (k_{w}*k_{h}*k_{d}+1)*C_{out}$$
> - $k_{w}$ = kernel width
> - $k_{h}$ = kernel height
> - $k_{d}$ = kernel depth *(= input depth)*
> - $C_{out}$ = number of filters
# Activation Function

In essence, a convolution operation produces a _**weighted sum**_ of pixel values. Therefore, it is a **linear** operation. Following a convolution with another will just be a convolution. 

![[ConvolutionSum.gif|center]]
>*Each element of the kernel is a **weight** that the network will **learn** during training.*

However, part of the reason CNNs are able to achieve such tremendous accuracies is **because of their non-linearity.** Non-linearity is necessary to produce **non-linear decision boundaries**, so that the output ***cannot be written as a linear combination of the inputs.*** If a non-linear activation function was not present, deep CNN architectures would devolve into a *single, equivalent convolutional layer,* which would not perform nearly as well.

That is why we follow the convolution with a [[Activation Functions#ReLU|ReLU activation]], which makes all negative values to zero.

![[ReLUFunction.png|center|500]]

>The ReLU activation function is specifically used as a *non-linear activation function*, as opposed to other non-linear functions such as _Sigmoid_ because it has been [empirically observed](https://arxiv.org/pdf/1906.01975.pdf "See page 29") that ***CNNs using ReLU are faster to train*** than their counterparts.

# Pooling Layer

> **Down-sampling** operation that reduces the dimensionality of the feature map.

Purpose of **gradually decreasing the spatial extent** of the network, which ***reduces the parameters and overall computation*** of the network.
![[MaxPooling.png|center]]
> *MaxPooling operation with a `2x2 kernel` with `(2,2) stride`. We can think of each 2 x 2 blocks as **pools** of numbers*.

Works like a convolution, but instead of computing the *weighted sum*, we return the **Maximum value (MaxPooling)** or the **Average value (AvgPooling)**. As such, **this layer doesn't have any trainable parameters.**

>[!note]- A little bit more in depth
> After applying the **ReLU** function, the feature map ends up with a lot of *'dead space'* that is, large areas containing only 0's. Having to carry these 0 activations through the entire network would **increase the size of the model without adding much useful information**. Instead, we would like to **_condense_** the feature map to retain only the most useful part -- the feature itself.
> 
> This in fact is what **maximum pooling** does. Max pooling takes a patch of activations in the original feature map and replaces them with the maximum activation in that patch.
>
>![[MaxPoolingCondense.png|center|400]]
>
>When applied after the ReLU activation, it has the effect of _'intensifying'_ features. The pooling step increases the proportion of active pixels to zero pixels.
>

>[!question]- Translation Invariance
>We called the zero-pixels *'unimportant'*. Does this mean they carry no information at all? In fact, the zero-pixels carry **positional information**. The blank space still positions the feature within the image. When `MaxPool2D` removes some of these pixels, it removes *some* of the positional information in the feature map. This gives a convnet a property called **translation invariance**. This means that a convnet with maximum pooling will tend **not to distinguish features by their location in the image.** (*Translation* is the mathematical word for changing the position of something without rotating it or changing its shape or size.)
>
>Watch what happens when we repeatedly apply maximum pooling to the following feature map.
>
>![[MaxPoolingTranslationInvariance1.png]]
>
>The two dots in the original image became *indistinguishable* after repeated pooling. In other words, pooling ***destroyed some of their positional information***. Since the network can no longer distinguish between them in the feature maps, it can't distinguish them in the original image either: it has become **invariant** to that difference in position.
>
>In fact, pooling only creates translation invariance in a network _over small distances_, as with the two dots in the image. Features that begin far apart will remain distinct after pooling; only _some_ of the positional information was lost, but not all of it.
>
>
>![](MaxPoolingInvariance2.png) 
>
>This invariance to small differences in the positions of features is a nice property for an image classifier to have. Just because of differences in perspective or framing, the same kind of feature might be positioned in various parts of the original image, but we would still like for the classifier to recognize that they are the same. Because this invariance is _built into_ the network, we can get away with using much less data for training: we no longer have to teach it to ignore that difference. This gives convolutional networks a big efficiency advantage over a network with only dense layers.
>

> [!abstract] Parameters
> *Only reduces dimension, no parameters to be learned.*

# Flatten Layer

> Converts a three-dimensional layer in the network into a **one-dimensional vector** to fit the input of a fully-connected layer for classification. Used after all Conv blocks so that we can fit our output to a ***fully connected layer.***

# Fully Connected Layer

> Traditional *feed-forward* neural network that take the high-level features learned by convolutional layers and use them for final predictions.

> [!abstract] Parameters
> $$Parameters = (input \; units*output \; units) + output \; units$$

# Regularization

> Deep learning models, especially CNNs, are particularly susceptible to **overfitting** due to their capacity for *high complexity* and their ability to learn detailed patterns in large-scale data.

**Overfitting** - model becomes too closely ***adapted to the training data***, capturing even its random fluctuations. The model describes features that arise from *noise* or *variance* in the data, rather than the **underlying distribution** from which the data were drawn.

![[UnderfittingVsOverfitting.png|center|600]]

Several regularization techniques can be applied to mitigate overfitting in CNNs :

- **Batch normalization:** The overfitting is reduced at some extent by normalizing the input layer by adjusting and scaling the activations. This approach is also used to **speed up and stabilize the training process.**
- **Dropout:** This consists of randomly *dropping* some neurons during the training process, which forces the remaining neurons to **learn new features** from the input data.
- **L1 and L2 normalizations:** Both L1 and L2 are used to add a **penalty to the loss function** based on the size of weights. More specifically, *L1 encourages the weights to be spare*, leading to better feature selection. On the other hand, L2 *(also called **weight decay**)* **encourages the weights to be small**, preventing them from having **too much influence** on the predictions.
- **Early stopping:** This consists of consistently monitoring the model’s performance on validation data during the training process and stopping the training whenever the validation error does not improve anymore.
- **Data augmentation:** This is the process of artificially increasing the size and diversity of the training dataset by applying random transformations like rotation, scaling, flipping, or cropping to the input images.
- **Noise injection:** This process consists of adding noise to the inputs or the outputs of hidden layers during the training to make the model more robust and prevent it from a weak generalization.
- **Pooling Layers:** This can be used to reduce the spatial dimensions of the input image to provide the model with an abstracted form of representation, hence reducing the chance of overfitting.

*More [[Regularization|here]].*

---
# 📂 Ressources

> 📖 <u><b>Articles</b></u>
>*[CNN Explainer](https://poloclub.github.io/cnn-explainer/)*
>*[Introduction to CNNs by datacamp](https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns)*
>*[Kaggle Computer Vision Course](https://www.kaggle.com/learn/computer-vision)*
>*[Comprehensive Guide to CNNs](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)*
>*[Convolution and ReLU](https://medium.com/@danushidk507/convolution-and-relu-fb69eb78dd0c)*
>*[Batch Norm Explained Visually](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)*
>
>📌<u><b>Additional</b></u>
>![[ConvolutionMovement.png|center|150]]
>>*Movement of a kernel.*
>
>❓ <u><b>Questions</b></u>
>*[Why do we make convolutions on RGB images?](https://datahacker.rs/convolution-rgb-image/)*
>*[Why we use activation function after convolution layer in Convolution Neural Network?](https://stackoverflow.com/questions/52003277/why-we-use-activation-function-after-convolution-layer-in-convolution-neural-net)*
>*[Reason behind performing dot product on Convolutional Neural networks](https://stats.stackexchange.com/questions/532630/reason-behind-performing-dot-product-on-convolutional-neural-networks)*
>*[What's the purpose of using a max pooling layer with stride 1 on object detection](https://stackoverflow.com/questions/56733596/whats-the-purpose-of-using-a-max-pooling-layer-with-stride-1-on-object-detectio)*