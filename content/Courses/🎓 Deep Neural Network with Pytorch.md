---
title: 🎓 Deep Neural Network with Pytorch
draft: false
tags:
---
> Deep Neural Networks with PyTorch. [🔗](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)

Course Overview:
1. Tensor and Datasets
2. Linear Regression
3. Linear Regression PyTorch
4. Multiple Input Output Linear Regression
	1. Multiple Linear Regression Prediction
	2. Multiple Output Linear Regression
5. Logistic Regression for Classification
	1. Prediction
	3. Cross Entropy Loss
6. Softmax
7. Neural Network
8. Deep Neural Networks
	1. Deep Neural Networks
	2. Dropout
	3. Weight initialization
	4. Gradient Descent with Momentum
	5. BatchNorm
9. CNNs
## 🧩 Code Snippets

### Layers

```python
nn.Linear(in_size, out_size)
nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
nn.MaxPool2d(kernel_size=2, stride=1)

nn.Dropout(p=p)
nn.BatchNorm2d(out_1)

torch.sigmoid(Layer)
torch.tanh(Layer)
torch.relu(Layer)
```

### Exemple using Sequential
```python
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.Sigmoid(),
    nn.Linear(hidden_dim, output_dim))
```

### Weights Initialization
```python
Optional : Weights Initialization (Nothing = Default method)
# He method
torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
# Xavier method
torch.nn.init.xavier_uniform_(linear.weight)
```
### Multi-layered model constructor

```python
class Net_He(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super(Net_He, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            # Optional : Weights Initialization (Nothing = Default method)
            # He method
            # torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            # Xavier method
            # torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(linear)

    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = F.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x
```

### CNN with BatchNorm

```python
class CNN_batch(nn.Module):
    
    # Contructor
    def __init__(self, out_1=16, out_2=32,number_of_classes=10):
        super(CNN_batch, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)

        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)

        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)
        self.bn_fc1 = nn.BatchNorm1d(10)
    
    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x=self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x=self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x=self.bn_fc1(x)
        return x
```
### GPU and Pre-trained models
**GPU in PyTorch**
```python
# Check if you have a working compatible GPU.
torch.cuda.is_available()
# Define GPU
device = torch.device('cuda:0')

# Use example
# The to method sends the tensor to the GPU and returns a tensor
# with a specify device.
torch.tensor([1,2,3]).to(device)
# >>> tensor([1,2,3], device='cuda:0')

# CNN with GPU
model = CNN()
model.to(device)
# During training phase, must send data to the GPU
for x,y in train_loader:
	x, y = x.to(device), y.to(device)
```


**TorchVision models**
Here is an example of how to use the pre-trained image classification models:
```python
# Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
# Transform method, which can be applied to DataLoaders
# The input data can come in a variety of sizes. They need to be normalized to a fixed size and format before batches of data are used together for training.
preprocess = weights.transforms()

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Changing the FC layer 
model.fc = nn.Linear(512,7) # 512 inputs, 7 classes
# or making it more complex
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 7)
)

# Optimizer
torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr=0.003)

# Training
model.train()

# Inference
model.eval()
```