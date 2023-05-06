# CS324 Assignment 2

## Part Ⅰ: PyTorch MLP

### Task 1

#### Model definition

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

#### Model train

```python
data, label = None, None
if opt.generator == 'moons':
    data, label = make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=opt.seed)
elif opt.generator == 'circles':
    data, label = make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=opt.seed)
elif opt.generator == 'iris':
    data, label = load_iris(return_X_y=True)
    label = (label > 0).astype(int)		# split dataset into only two classes
x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.8, random_state=0)
```

Above is the procedure of data preparation. First generate corresponding dataset according to program args, and split them into train datasets and test datasets. To be noticed in **iris** dataset preparation, in order to be able to classify iris datasets in numpy's mlp implementation as well, I have treated iris datasets into two classes: class with label 0 and class with label greater than 0. Also, I add an argument `--seed` to specialize the random seed (`random_state`) when using dataset generator in `sklearn.datasets`. This can help me to make same datasets for both numpy version and pytorch version of MLP implementation.

Both numpy and pytorch implementation uses **SGD** optimizer.

### Task 2

#### Hyper-parameter setting

| Parameter     | Value        |
| ------------- | ------------ |
| hidden_units  | '20'         |
| learning_rate | 0.01         |
| epochs        | 1500         |
| batch_size    | / (no batch) |

- pytorch implementation

  <center>
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304210914006.png" alt="image-20230421091432928" style="zoom:50%;" />
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304210915442.png" alt="image-20230421091511387" style="zoom:30%;" />
  </center>

- numpy implementation

  <center>
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304210917073.png" alt="image-20230421091704047" style="zoom:50%;" />
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304210917781.png" alt="image-20230421091729727" style="zoom:30%;" />
  </center>

#### Datasets

The datasets below all use the same MLP structure (with hidden dim 20)

- Moon datasets:

  Final accuracy of numpy(left) and pytorch(right) implementation of MLP on moon datasets are **0.95** and **1.0**.

  <center>
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061120479.png" alt="image-20230406112044454" style="zoom:60%;" />
  	<img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061120015.png" alt="image-20230406112020986" style="zoom:60%;" />
  </center>
  The datasets looks like this.
  
  <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061129897.png" alt="image-20230406112900869" style="zoom:60%;" />
  
  As the curves show, the final accuracy of numpy version MLP is a little bit smaller than that of pytorch implementation. I haven't a reasonable cause of such situation. I guess this is due to the difference of gradient computation and optimizer implementation.

- Circle datasets:

  I have also test my MLP implementation accuracy on `make_circles()` datasets. The results are listed as below.

  Final accuracy of numpy and pytorch implementation of MLP on moon datasets are **0.92** and **0.96**.

  <center>
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061210265.png" alt="image-20230406121024237" style="zoom:60%;" />
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061210225.png" alt="image-20230406121040197" style="zoom:60%;" />
  </center>
  
  As the picture below shows, the decision boundary is a close oval-like curve between the blue points and red points, which is not linear. However the moon datasets can be divided into two classes by a linear decision boundary (a straight line) with a relatively high accuracy. So circle dataset is harder than moon datasets to be classified by the same MLP model, which can explain why the accuracy on circle datasets is lower than that on moon datasets.
  
  <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304061210569.png" alt="image-20230406121059538" style="zoom:60%;" />

- Iris datasets:

  Final accuracy of numpy and pytorch implementation of MLP on moon datasets are both **1.0**.

  <center>
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211241671.png" alt="image-20230421124158643" style="zoom:60%;" />
      <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211242787.png" alt="image-20230421124240759" style="zoom:60%;" />
  </center>

  As the picture below shows, it is easy to find a linear decision boundary (a straight line) to divide the datasets into two classes. This iris datasets are the easiest one to be classified. So we get the highest accuracy on the iris datasets.

  <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211243677.png" alt="image-20230421124313648" style="zoom:60%;" />

### Task 3

> Note: hyper-parameter setting of below model training procedure is the same as that in Task 2

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
  
  <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211051907.png" alt="image-20230421105144823" style="zoom:60%;" />
  
  As the result shows, the difference between the training accuracy and the test accuracy is large. The final train accuracy is **0.98** and test accuracy is **0.46**.This indicates that serious overfitting occurred during the training process. This is because the model is simple and has poor generalization ability. The model may over-match the details of the training data and ignore the general patterns, resulting in poor performance on new data.
  
  
  
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
              nn.ReLU()
          )
          self.fc = nn.Linear(64, 10, bias=True)
  ```
  
  <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304211139981.png" alt="image-20230421113923949" style="zoom:60%;" />

  The final train accuracy and test accuracy are **0.99** and **0.55**. The test accuracy has improved compared to the previous simple MLP model, which indicates that the generalization of the model can be improved to some extent by simply increasing the number of layers of the model to make it complex. However, the over-fitting problem still exists.

  Note: After that I also tried to increase the depth of the model, but in the end they all showed obvious, similar overfitting phenomena as above, so I won't repeat them here
  
- Avoid over-fitting: I try to take some methods to improve the overfitting phenomenon of the model, here are the improvement methods and results.

  - **Dropout**: Dropout is a commonly used regularization method that reduces redundancy in the model and prevents overfitting by randomly turning off some neurons during the training process. During each training session, some neurons are randomly selected to be turned off, so that each neuron cannot be too dependent on some specific input. This process can somewhat prevent the model from learning some noise or random features into the model as well, thus improving the overfitting phenomenon.

    ```python
    class MLP(nn.Module):
        def __init__(self, dropout_rate=0.5):
            super(MLP, self).__init__()
            self.hidden = nn.Sequential(
                nn.Linear(3072, 2048, bias=True),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),			# Randomly added Dropout layer
                nn.Linear(2048, 1024, bias=True),
                nn.ReLU(),
                nn.Linear(1024, 256, bias=True),
                nn.Dropout(p=dropout_rate),			# Randomly added Dropout layer
                nn.ReLU(),
                nn.Linear(256, 128, bias=True),
                nn.ReLU(),
                nn.Linear(128, 64, bias=True),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)			# Randomly added Dropout layer
            )
            self.fc = nn.Linear(64, 10, bias=True)
    ```

    Set dropout rate to 0.5. The results are listed below.

    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271047391.png" alt="image-20230427104700363" style="zoom:60%;" />

    The final train accuracy and test accuracy are **0.547** and **0.492**. As the results show, after adding the `Dropout` layer, the overfitting phenomenon is greatly suppressed, but the corresponding model accuracy is also reduced. This is because `Dropout` randomly discards some neurons in each training iteration, which prevents the model from fully learning the correct features from the training data and therefore makes the training error increase.

  - **L2 Regularization**: 

    The training part of the code has been slightly changed

    ```python
    outputs = mlp(inputs)
    
    # Compte L2 regulatization loss
    l2_lambda = 0.01
    regularization_loss = 0
    for param in mlp.parameters():
    	regularization_loss += torch.norm(param, 2)
        
    loss = criterion(outputs, labels) + l2_lambda * regularization_loss
    loss.backward()
    optimizer.step()
    ```

    The result

    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271135511.png" alt="image-20230427113526477" style="zoom:60%;" />

    The final train accuracy and test accuracy are **0.770** and **0.520**. As the result shows, the difference between train accuracy and test accuracy has decreased, which means L2 regularization do avoid over-fitting to some extent. L2 regularization limits the complexity of the model by penalizing the sum of squares of the model weights. In the optimization process, we usually want to minimize the loss function while minimizing the sum of squares of the model weights, which is equivalent to a certain reduction of the model weights. This has the advantage of effectively reducing the complexity of the model and preventing the model from being overfitted on the training set.

According to the experiment results above, it can be inferred that simple MLP network can not handle CIFAR-10 dataset classification task well.

#### Further improvement

Under the guidance of Zhichen Liu, I tried to apply a new MLP structure to complete the classification task of the cifar-10 dataset.

<img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305022306463.png" alt="未命名绘图.drawio" style="zoom:60%;" />

The model structure is demonstrated as above. The model is inspired from convolutional neural network. CNN can extract information from original inputs by reducing input width and height, and mapping the information into higher dimension (more output channels). 

```python
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlpx = nn.Sequential(
            nn.Linear(96, 256),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU()
        )
        self.mlpy = nn.Sequential(
            nn.Linear(96, 256),
            nn.ReLU(),
            nn.Linear(256, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 10, bias=True)
        )
        
    def forward(self, input):
        x = torch.cat((input[:, 0, :, :], input[:, 1, :, :], input[:, 2, :, :]), dim=1)
        x = torch.einsum('ijk->ikj', x)		# transpose width and height
        y = torch.cat((input[:, 0, :, :], input[:, 1, :, :], input[:, 2, :, :]), dim=2)
        x = self.mlpx(x)
        y = self.mlpy(y)
        out = torch.cat((x, y), dim=1)
        out = out.view(-1, 2048)			# flatten
        out = self.fc(out)
        return out
```

`mlpx` and `mlpy` in the model extract information along x and y axis separately, compressing the information therein into a 32x32 matrix. Afterwards, their results are stitched together and the subsequent computation of the fully connected MLP layer is performed.

After 100 epochs, the model gets a result with train accuracy **0.962** and test accuracy **0.634**.

<img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305022250369.png" alt="image-20230502225016289" style="zoom:60%;" />

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

Suppose each channel of conv layer's input is $H_{in} \times W_{in} \times C_{in}$ (height x width x channel_num), with $n_k$ conv kernels $k \times k \times C_{in}$, stride $s$ and padding $p$. Thus we can get the output size of this channel can be computed by formula.
$$
(\frac{H_{in}-k+2p}{s}+1, \frac{W_{in}-k+2p}{s}+1, n_k)
$$
Here $n_k$ is the output channel number.

The output size of maxpool layer can be computed by the same formula above.



Since each “conv” layer is composed of:  **2D-CONV + Bath Normalisation +  ReLU**. So we first define such "conv" layer.

```python
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      in_channels=in_channels,
                      out_channels=out_channels),	# Convolution operation
            nn.BatchNorm2d(out_channels),			# Batch normalization
            nn.ReLU(inplace=True)					# ReLU activation function
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
            Linear(in_features=512, out_features=n_classes),	# Fully connected layer
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
    
    

  According to the curves shown, the use of different optimizers (`Adam`, `NAdam`, `RMSprop`) has little or no effect on the trend of loss and accuracy variation, and the final accuracy during training.

- Learning rate comparison (with Adam optimizer, batch_size=128, epochs=100)

  - `lr=0.0001`

    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304270939935.png" alt="image-20230427093855854" style="zoom:60%;" />

  - `lr=0.001`

    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304270939159.png" alt="image-20230427093927130" style="zoom:60%;" />

  - `lr=0.01`

    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304270939140.png" alt="image-20230427093948110" style="zoom:60%;" />

  - `lr=0.04`

    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271023959.png" alt="image-20230427102350929" style="zoom:60%;" />

  According to the curves corresponding to different learning rates, it can be found that the difference between training accuracy and test accuracy, as well as between training loss and test loss, decreases gradually (at a fixed training size) as the learning rate increases within a certain range. This is because as the learning rate increases, the larger the step we update the parameter weights each time during the training process, which means the faster the approach to the optimal solution, and therefore the difference between the training and testing accuracies and the loss will gradually decrease.

  However, if the learning rate is too large, the final result of the model will not be better. This is because when the learning rate is too large, the model updates the parameter weights in too large a step each time and easily crosses the optimal solution, so there will be a situation like the above where the accuracy and loss remain constant at `lr=0.04`.



## Part Ⅲ: PyTorch RNN

### Task 1

```python
class VanillaRNN(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Whx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Whh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wph = nn.Linear(hidden_dim, output_dim, bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(self.batch_size, self.seq_length, self.input_dim)
        h = torch.zeros(self.batch_size, self.hidden_dim).cuda() # Initialise hidden state to zero vector
        for t in range(self.seq_length):
            xt = x[:, t, :]								# Get input at time step t
            h = self.tanh(self.Whx(xt) + self.Whh(h))	# Compute hidden state
        o = self.Wph(h)			# Compute output
        y = self.softmax(o)
        return y
```

$$
\begin{array}{cc}
h^{(t)} = \tanh (W_{hx} x^{(t)} + W_{hh} h^{(t-1)} + b_h) \\
o^{(t)} = W_{ph} h^{(t)} + b_o \\
\widetilde{y}^{(t)} = softmax(o^{(t)})
\end{array}
$$

The model does not explicitly define $b_h$ and $b_o$. This is because `nn.Linear` comes with bias and can be made to include bias by setting `bias=True` (`self.Whh` and `self.Wph` includes bias operation). In the `forward` function, a for-loop with `self.seq_length` iteration is used to compute the hidden states along $t$ time steps, i.e. along with the input sequence. And then compute final output `o`, also with a softmax operation to get the probability of each choice.

During each train iteration, I will execute the code below first.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.max_norm)
```

The code above is to prevent gradient explosion during training. Gradient clipping solves the problem of exploding gradients by scaling down the gradient values to ensure that they do not exceed a pre-defined threshold. Specifically, if the norm of the gradients is greater than the threshold, then the gradients are rescaled so that the norm becomes equal to the threshold. This prevents the gradients from growing too large and helps to stabilize the training process.

### Task 2

#### Default `seq_length=5`

<img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271128549.png" alt="image-20230427112841511" style="zoom:60%;" />

When seq_length=5 and after training 2000 iter, the model accuracy reaches 1, which means the model is implemented correctly.

#### `seq_length` from 2 to 10

<center>
    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271151954.png" alt="image-20230427115114918" style="zoom:60%;" />
    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271151021.png" alt="image-20230427115127985" style="zoom:60%;" />
    <img src="C:/Users/DV299/AppData/Roaming/Typora/typora-user-images/image-20230427115147711.png" alt="image-20230427115147711" style="zoom:60%;" />
    <img src="C:/Users/DV299/AppData/Roaming/Typora/typora-user-images/image-20230427115424786.png" alt="image-20230427115424786" style="zoom:60%;" />
    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271152691.png" alt="image-20230427115218658" style="zoom:60%;" />
    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271152922.png" alt="image-20230427115240883" style="zoom:60%;" />
    <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271153226.png" alt="image-20230427115349196" style="zoom:60%;" />
</center>

It can be observed from the above results that as the `seq_length ` increases, the more flat the accuracy curve is in the initial stage of training, or the more difficult it is for the accuracy (loss) to jump out of a poor value and thus produce significant changes. This is because when the sequence is long, those inputs that are located at earlier time steps have less impact in the final output, i.e., they are less likely to be "remembered" by the model, which is also known as the gradient vanishing problem.

#### `seq_length=10,15,20`

<img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271213763.png" alt="image-20230427121336728" style="zoom:60%;" />

<img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271213201.png" alt="image-20230427121349167" style="zoom:60%;" />

<img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202304271214800.png" alt="image-20230427121404765" style="zoom:60%;" />

With 5000 iters trained, the accuracy of the model increased significantly only when the seq_length was 5 and finally reached 1. This also confirms the above speculation that the longer the seq_length, the more difficult it is for the model to make significant changes, i.e., the more difficult it is to "learn" something.

