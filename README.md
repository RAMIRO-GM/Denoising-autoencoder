# Denoising-autoencoder
Denoising convolutional autoencoder in Pytorch.

## Encoder:
Series of 2D convolutional and max pooling layers. Using Relu activations.

## Decoder:
Series of 2D transpose convolutional layers. Using Relu activations.


The architecture is the following:
```python
ConvDenoiser(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (t_conv1): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2))
  (t_conv2): ConvTranspose2d(32, 64, kernel_size=(2, 2), stride=(2, 2))
  (t_conv3): ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2))
  (conv_out): Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
```
### Optimizer
Adam Optimizer, alpha and beta values: default values. Learning rate:.001
**Loss: ** MSE, Mean Squared Error 
