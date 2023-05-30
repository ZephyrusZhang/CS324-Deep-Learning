# CS324 Assignment 3 Report

## Part I: PyTorch LSTM

### Task 1

```python
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Initialization here ...
        self.Wgx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Wgh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wix = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Wih = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wfx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Wfh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wox = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Woh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wph = nn.Linear(hidden_dim, output_dim, bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Implementation here ...
        batch_size, seq_length = x.size(0), x.size(1)
        x = x.view(batch_size, seq_length, self.input_dim)
        h_t = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t = torch.zeros(batch_size, self.hidden_dim).cuda()
        for t in range(seq_length):
            xt = x[:, t, :]
            g_t = self.tanh(self.Wgx(xt) + self.Wgh(h_t))
            i_t = self.sigmoid(self.Wix(xt) + self.Wih(h_t))
            f_t = self.sigmoid(self.Wfx(xt) + self.Wfh(h_t))
            o_t = self.sigmoid(self.Wox(xt) + self.Woh(h_t))
            c_t = g_t * i_t + c_t * f_t
            h_t = self.tanh(c_t) * o_t
        p_t = self.Wph(h_t)
        y_t = self.softmax(p_t)
        return y_t
```

$$
\begin{array}{cc}
g^{(t)} = tanh (W_{gx}x^{(t)} + W_{gh}h^{(t-1)} + b_g) \\
i^{(t)} = \sigma (W_{ix}x^{(t)} + W_{ih}h^{(t-1)} + b_i) \\
f^{(t)} = \sigma (W_{fx}x^{(t)} + W_{fh}h^{(t-1)} + b_f) \\
o^{(t)} = \sigma (W_{ox}x^{(t)} + W_{oh}h^{(t-1)} + b_o) \\
c^{(t)} = g^{(t)} \odot i^{(t)} + c^{(t-1)} \odot  f^{(t)} \\
h^{(t)} = tanh (c^{(t)}) \odot o^{(t)} \\
p^{(t)} = (W_{ph}h^{(t)} + b_p) \\
\widetilde{y}^{(t)} = softmax(p^{(t)})
\end{array}
$$

The model does not explicitly define bias $b_g$, $b_i$, $b_f$ , $b_o$, and $b_p$. This is because `nn.Linear` comes with bias and can be made to include bias by setting `bias=True` (`self.Whh` and `self.Wph` includes bias operation). In the `forward` function, a for-loop with `self.seq_length` iteration is used to compute the hidden states along $t$ time steps, i.e. along with the input sequence. And then compute final output `p_t`, also with a softmax operation to get the probability of each choice.

During each train iteration, I will execute the code below first.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.max_norm)
```

The code above is to prevent gradient explosion during training. Gradient clipping solves the problem of exploding gradients by scaling down the gradient values to ensure that they do not exceed a pre-defined threshold. Specifically, if the norm of the gradients is greater than the threshold, then the gradients are rescaled so that the norm becomes equal to the threshold. This prevents the gradients from growing too large and helps to stabilize the training process.

### Task 2

I made a comparison between RNN and LSTM. Here is their accuracy and loss curves.

| Sequence Length |                             RNN                              |                             LSTM                             |
| :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|       10        | <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305292158611.png" alt="image-20230529215825216" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305292215360.png" alt="image-20230529221552330" style="zoom:50%;" /> |
|       15        | <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305292159719.png" alt="image-20230529215902691" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305292212438.png" alt="image-20230529221253405" style="zoom:50%;" /> |
|       20        | <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305292159799.png" alt="image-20230529215913773" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305292223767.png" alt="image-20230529222331736" style="zoom:50%;" /> |

Due to the definition of `PalindromDataset`, I need to adjust the two hyper-parameters of RNN and LSTM training process, to try my best to control variables to be the same (控制变量), and make the comparison convincing.

```python
# Train loop of RNN
for step, (batch_inputs, batch_targets) in enumerate(train_loader):
    # Code
    if step == OPT.train_steps:
            break
```

```python
# Train loop of LSTM
for epoch in range(OPT.max_epoch):
        for step, (batch_inputs, batch_targets) in enumerate(train_loader):
            # Code
```

Above are train loop code segments of RNN and LSTM. As you can see they are different. To ensure that the model is trained with an equal amount of data during training, I set hyper-parameters as below (`max_norm` is referred to max norm of the gradients in gradient clip).

```shell
# RNN
OPT.learning_rate = 0.001
optimizer: RMSprop
OPT.train_steps = 5000
OPT.batch_size = 128
OPT.max_norm = 10

# LSTM
OPT.learning_rate = 0.001
optimizer: RMSprop
OPT.max_epoch = 50
OPT.data_size = 128000
OPT.batch_size = 128
OPT.max_norm = 10
```

In RNN training, back propagation and weight update process will be executed $5000$ times. And in LSTM training, back propagation and weight update process will be executed $50 * (12800 / 128) = 5000$ times.



As the figures in table shows,

- `seq_length=10`: The accuracy of the RNN starts to increase significantly around 3800 steps and reaches 100% at around 4600 steps. The accuracy of the LSTM starts to increase significantly around 0.8*100=80 steps and converges to 100% at around 31\*100=3100 steps.
- `seq_length=15`: The accuracy of the RNN has consistently been low.  The accuracy of the LSTM starts to increase significantly around 0.8*100=80 steps and converges to100% at around 41\*100=4100 steps.
- `seq_length=20`: The accuracy of the RNN has consistently been low.  The accuracy of the LSTM starts to increase significantly around 0.8*100=80 steps and reaches 100% at around 34\*100=3400 steps. It has not converged to 100% within 5000 steps.

The experimental results mentioned above indicate that the LSTM implementation in this assignment is correct and validate that LSTM indeed outperforms regular RNN in handling long sequences.



I also test my LSTM on even longer sequence.

| Sequence Length |                   Accuracy and Loss Curve                    |
| :-------------: | :----------------------------------------------------------: |
|       25        | <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305301144950.png" alt="image-20230530114359863" style="zoom:50%;" /> |
|       30        | <img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305301211045.png" alt="image-20230530121134010" style="zoom:50%;" /> |

For longer sequence, the `PalindromeDataset` will report `OverflowError: Python int too large to convert to C long`. So they are not included in this report.

## Part II: Generative Adversarial Networks

### Task 1

![dcgan_generator](https://pytorch.org/tutorials/_images/dcgan_generator.png)

The GAN I use in this assignment is DCGAN. Its model definition is listed as below

```
Generator
    ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ReLU(inplace=True)
    ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ReLU(inplace=True)
    ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ReLU(inplace=True)
    ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    Tanh()
```

```
Discriminator:
    Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    LeakyReLU(negative_slope=0.2, inplace=True)
    Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    LeakyReLU(negative_slope=0.2, inplace=True)
    Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    LeakyReLU(negative_slope=0.2, inplace=True)
    Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    Sigmoid()
```

Conv2dTranspose, also known as transposed convolution or deconvolution, is used to perform the transpose operation of two-dimensional convolution. It allows expanding the spatial dimensions of the input tensor and is commonly used for upsampling images or in the deconvolution layers of generative models. This operation can be seen as the inverse of convolution.

We know that, if input dimension of convolution is $n$, kernel size is $k$, padding is $p$, stride is $s$, then the output size is $\lfloor \frac{n-k+2p}{s}+1 \rfloor$. Thus we can know that the output size of convolution transpose is $s(n-1)-2p+k$. If the input size of Generator is $(100, 1, 1)$. Size of each layer is

|  Layer (type)   | Output Shape (channel, width, height) |
| :-------------: | :-----------------------------------: |
| ConvTranspose2d |              (512, 4, 4)              |
|   BatchNorm2d   |              (512, 4, 4)              |
|      ReLU       |              (512, 4, 4)              |
| ConvTranspose2d |              (256, 7, 7)              |
|   BatchNorm2d   |              (256, 7, 7)              |
|      ReLU       |              (256, 7, 7)              |
| ConvTranspose2d |             (128, 14, 14)             |
|   BatchNorm2d   |             (128, 14, 14)             |
|      ReLU       |             (128, 14, 14)             |
| ConvTranspose2d |              (1, 28, 28)              |
|      Tanh       |              (1, 28, 28)              |

Which means the generated image size is correct, same as MNIST datasets.



The training process is divided into 2 parts: train generator and train discriminator.

```python
# Train Discriminator
output = netD(real_img)
label = torch.full((batch_size,), 1.0, device=DEVICE)  # real label
lossD_real = criterion(output, label)
lossD_real.backward()

noise = torch.randn(batch_size, OPT.latent_dim, 1, 1, device=DEVICE)
fake_img = netG(noise)
label.fill_(0.0)  # fake label
output = netD(fake_img.detach())
lossD_fake = criterion(output, label)
lossD_fake.backward()

lossD = lossD_real + lossD_fake
optimizerD.step()
```

During the training of the discriminator, it is necessary to perform separate forward passes and loss calculations for both real data and fake data generated by the generator (their labels being 1 and 0, respectively). Finally, the accumulated gradients from backpropagation are used to update the weights.

```python
# Train Generator
netG.zero_grad()
label.fill_(1.0)
output = netD(fake_img)
lossG = criterion(output, label)
lossG.backward()
optimizerG.step()
```

During the training of the generator, since we aim to deceive the discriminator, the label is set to 1. We perform gradient backpropagation and update based on this label to train the generator.

### Task 2

Hyper-parameters setting:

```
batch_size=64
max_epochs=100
learning_rate=0.0002
latent_dim=100
```

<img src="https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305301225869.png" alt="image-20230530122559836" style="zoom:80%;" />

As the loss curve above shows, the loss curves of both the discriminator and the generator are oscillating throughout the training. Moreover, from a general distribution perspective, when the generator's loss decreases, the discriminator's loss also decreases, and when the generator's loss increases, the discriminator's loss decreases. This indicates that the generator and discriminator are indeed "adversarial" to each other: the generator attempts to deceive the discriminator (corresponding to an increase in discriminator's loss and a decrease in generator's loss), while the discriminator tries to distinguish the fake images generated by the generator (leading to a decrease in discriminator's loss and an increase in generator's loss).

|                           Epoch 1                            |                           Epoch 50                           |                          Epoch 100                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20230529234405797](https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305292344824.png) | ![image-20230529234410845](https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305292344869.png) | ![image-20230529234422828](https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305292344854.png) |

### Task 3

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = Generator(100, 128, 1).to(device)
generator.load_state_dict(torch.load('./model/mode.pth', map_location=device))
noises = []
start = torch.randn(1, 100, 1, 1, device=device)
end = torch.randn(1, 100, 1, 1, device=device)
for i in range(9):
    noises.append(start + (end - start) / 8 * i)

fig, axs = plt.subplots(1, 9, figsize=(28, 28))
for i in range(len(noises)):
    img = generator(noises[i]).squeeze().cpu().detach().numpy()
    axs[i].set_axis_off()
    axs[i].imshow(img)
plt.show()
```

![image-20230529234811514](https://raw.githubusercontent.com/zephyrszwc/zephyrs-image/master/202305292348551.png)

Based on the results, the two noise inputs generated images of 4 and 6, respectively. By interpolating between these two noise inputs and generating corresponding images, it can be observed that the images are gradually transitioning from 4 to 6.