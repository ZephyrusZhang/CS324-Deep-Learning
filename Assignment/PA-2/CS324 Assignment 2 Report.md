# CS324 Assignment 2

## Part Ⅰ: PyTorch MLP

### Task 1

Just like Assignment 1, we need to implement a structure as the following figure shows

<img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061053510.png" alt="image-20230406105300454" style="zoom: 40%;" />

What we need to do is to stack `nn.Linear` and `nn.ReLU` repeatedly (according to the input parameter `n_hidden`), and do a **softmax** to get the predicted probability of each class.

```python
class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_classes):
        super().__init__()
        self.model = nn.Sequential()
        units = [n_inputs] + n_hidden + [n_classes]
        for i in range(len(n_hidden)):
            self.model.append(nn.Linear(units[i], units[i + 1]))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(units[-2], units[-1]))
        self.model.append(nn.Softmax(dim=1))
```

Here we use `nn.Sequential` to stack each layer, which will convenient **forward** process.

Besides the weight initialization of numpy implementation is changed to Xavier initialization, which is the same as the weight initialization in pytorch. Both weight and bias will be initialized as $U(-\sqrt{k}, \sqrt{k})$, where $U$ is standard normalization distribution and $k = \frac{1}{\text{in\_features}}$.

```python
class Linear(object):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        std = np.sqrt(1.0 / in_features)
        self.params = {
            'weight': np.random.normal(-std, std, size=(out_features, in_features)),
            'bias': np.random.normal(-std, std, size=out_features)
        }
```

### Task 2

I add an argument `--seed` to specialize the random seed (`random_state`) when using dataset generator in `sklearn.datasets`. This can help me to make same datasets for both numpy version and pytorch version of MLP implementation.

```python
# Specialize random seed
make_moons(..., random_state=opt.seed)
make_circles(..., random_state=opt.seed)
```

Using **SGD** as optimizer. The accuracy and loss on both train and test datasets of both numpy(left) and pytorch(right) implementation are listed below.



#### Datasets

- Moon datasets:

  Final accuracy of numpy and pytorch implementation of MLP on moon datasets are **1.0** and **0.95**.

  <center>
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061120479.png" alt="image-20230406112044454" style="zoom:80%;" />
  	<img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061120015.png" alt="image-20230406112020986" style="zoom:80%;" />
  </center>

  The datasets looks like this.

  <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061129897.png" alt="image-20230406112900869" style="zoom:80%;" />

  As the curves show, the final accuracy (after 1500 epochs) of numpy version MLP is a little bit smaller than that of pytorch implementation. I haven't a reasonable cause of such situation. I guess this is due to the difference of gradient computation and optimizer implementation.

- Circle datasets:

  I have also test my MLP implementation accuracy on `make_circles()` datasets. The results are listed as below.

  Final accuracy of numpy and pytorch implementation of MLP on moon datasets are **0.96** and **0.92**.

  <center>
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061210265.png" alt="image-20230406121024237" style="zoom:80%;" />
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061210225.png" alt="image-20230406121040197" style="zoom:80%;" />
  </center>

  <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061210569.png" alt="image-20230406121059538" style="zoom:80%;" />

- Iris datasets:

  Final accuracy of numpy and pytorch implementation of MLP on moon datasets are both **1.0**.

  <center>
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211241671.png" alt="image-20230421124158643" style="zoom:80%;" />
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211242787.png" alt="image-20230421124240759" style="zoom:80%;" />
  </center>

  <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211243677.png" alt="image-20230421124313648" style="zoom:80%;" />

  Note: In order to be able to classify iris datasets in numpy's mlp implementation as well, I have treated iris datasets into two classes: class with label 0 and class with label greater than 0.

#### Hyper-parameter setting

- pytorch implementation

  <center>
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304210914006.png" alt="image-20230421091432928" style="zoom:60%;" />
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304210915442.png" alt="image-20230421091511387" style="zoom:40%;" />
  </center>

- pytorch implementation

  <center>
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304210917073.png" alt="image-20230421091704047" style="zoom:60%;" />
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304210917781.png" alt="image-20230421091729727" style="zoom:40%;" />
  </center>

### Task 3

- Initial MLP model setting

  ```python
  class MLP(nn.Module):
      def __init__(self):
          super(MLP, self).__init__()
  
          self.hidden = nn.Sequential(
              nn.Linear(3072, 128, bias=True),
              nn.ReLU(),
              nn.Linear(128, 64, bias=True),
              nn.ReLU()
          )
          self.fc = nn.Linear(64, 10, bias=True)
  
      def forward(self, x):
          x = x.view(-1, 3072)
          x = self.hidden(x)
          x = self.fc(x)
          return x
  ```

  The result:

  <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211051907.png" alt="image-20230421105144823" style="zoom:70%;" />

  As the result shows, the difference between the training accuracy and the test accuracy is large. The final train accuracy is 0.98 and test accuracy is 0.46.This indicates that serious overfitting occurred during the training process. This is because the model is simpler and has poor generalization ability. The model may over-match the details of the training data and ignore the general patterns, resulting in poor performance on new data.

  

- Now we try to make our model more complex and see what will happen.

  ```python
  class MLP(nn.Module):
      def __init__(self):
          super(MLP, self).__init__()
  
          self.hidden = nn.Sequential(
              nn.Linear(3072, 2048, bias=True),
              nn.ReLU(),
              nn.Linear(2048, 1024, bias=True),
              nn.ReLU(),
              nn.Linear(1024, 256, bias=True),
              nn.ReLU(),
              nn.Linear(256, 128, bias=True),
              nn.ReLU(),
              nn.Linear(128, 64, bias=True),
              nn.ReLU(),
              nn.Dropout(p=dropout_rate)
          )
          self.fc = nn.Linear(64, 10, bias=True)
  ```

  <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211139981.png" alt="image-20230421113923949" style="zoom:70%;" />

  The final train accuracy and test accuracy are **0.99** and **0.55**. The test accuracy has improved compared to the previous simple MLP model, which indicates that the generalization of the model can be improved to some extent by simply increasing the number of layers of the model to make it complex. However, the over-fitting problem still exists.

## Part Ⅱ: PyTorch CNN

### Task 1

The structure of VGG network is listed below. And each conv layer is composed **2D-CONV + Bath Normalisation +  ReLU**

```
1. conv layer: k=3x3, s=1, p=1, in_channels=3, out_channels=64
2. maxpool: k=3x3, s=2, p=1, in_channels=64, out_channels=64
3. conv layer: k=3x3, s=1, p=1, in_channels=64, out_channels=128
4. maxpool: k=3x3, s=2, p=1, in_channels=128, out_channels=128
5. conv layer: k=3x3, s=1, p=1, in_channels=128, out_channels=256
6. conv layer: k=3x3, s=1, p=1, in_channels=256, out_channels=256
7. maxpool: k=3x3, s=2, p=1, in_channels=256, out_channels=256
8. conv layer: k=3x3, s=1, p=1, in_channels=256, out_channels=512
9. conv layer: k=3x3, s=1, p=1, in_channels=512, out_channels=512
10. maxpool: k=3x3, s=2, p=1, in_channels=512, out_channels=512
11. conv layer: k=3x3, s=1, p=1, in_channels=512, out_channels=512
12. conv layer: k=3x3, s=1, p=1, in_channels=512, out_channels=512
13. maxpool: k=3x3, s=2, p=1, in_channels=512, out_channels=512
14. linear, in_channels=512, out_channels=10
```

Suppose each channel of conv layer's input is $H_{in} \times W_{in} \times C_{in}$ (height x width x channel_num), with $n_k$ conv kernels $k \times k \times C_{in}$, stride $s$ and padding $p$. Thus we can get the output size of this channel can be computed by formula
$$
(\frac{H_{in}-k+2p}{s}+1, \frac{W_{in}-k+2p}{s}+1, n_k)
$$
Here $n_k$ is the output channel number.

The output size of maxpool layer can be computed by the same formula above.



Since each “conv” layer is composed of:  **2D-CONV + Bath Normalisation +  ReLU**. So we first define such "conv" layer.

```python
class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(ConvBNReLU, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      in_channels=in_channels,
                      out_channels=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)
```

Then according to the definition of VGG in slide, the model can be easily defined.

```python
class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=n_channels, out_channels=64),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=64, out_channels=128),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=128, out_channels=256),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=256, out_channels=256),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=256, out_channels=512),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=512, out_channels=512),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=512, out_channels=512),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=512, out_channels=512),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Sequential(
            Linear(in_features=512, out_features=n_classes),
            Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512)		# Flatten the 2D input for fully connected layer computation
        x = self.fc(x)
        return x
```



#### Result

- Optimizer comparison (with learning_rate=1e-4, batch_size=128, epochs=100)

  - `Adam`

    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211138324.png" alt="image-20230421113850289" style="zoom:60%;" />

    Final train accuracy and test accuracy **0.980** and **0.825**. 

  - `NAdam`

    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211224360.png" alt="image-20230421122350104" style="zoom:60%;" />

    Final train accuracy and test accuracy **0.980** and **0.824**.

  - `RMSprop`

    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211253246.png" alt="image-20230421125312217" style="zoom:60%;" />

    Final train accuracy and test accuracy **0.979** and **0.827**.

- Learning rate comparison (with Adam optimizer, batch_size=128, epochs=100)
