# Assignment 2: Training and Visualizing a CNN on CIFAR


CUDA is not available.  Training on CPU ...

Net(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1024, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=10, bias=True)
  (dropout): Dropout(p=0.25, inplace=False)
)

Epoch: 1 	Training Loss: 1.458343 	Validation Loss: 1.182367
Validation loss decreased (inf --> 1.182367).  Saving model ...
Epoch: 2 	Training Loss: 1.094803 	Validation Loss: 1.013165
Validation loss decreased (1.182367 --> 1.013165).  Saving model ...
Epoch: 3 	Training Loss: 0.938799 	Validation Loss: 0.945466
Validation loss decreased (1.013165 --> 0.945466).  Saving model ...
Epoch: 4 	Training Loss: 0.848077 	Validation Loss: 0.840706
Validation loss decreased (0.945466 --> 0.840706).  Saving model ...
Epoch: 5 	Training Loss: 0.775157 	Validation Loss: 0.844030
Epoch: 6 	Training Loss: 0.715195 	Validation Loss: 0.819683
Validation loss decreased (0.840706 --> 0.819683).  Saving model ...
Epoch: 7 	Training Loss: 0.659644 	Validation Loss: 0.825711
Epoch: 8 	Training Loss: 0.625043 	Validation Loss: 0.792922
Validation loss decreased (0.819683 --> 0.792922).  Saving model ...
Epoch: 9 	Training Loss: 0.586021 	Validation Loss: 0.787273
Validation loss decreased (0.792922 --> 0.787273).  Saving model ...
Epoch: 10 	Training Loss: 0.553661 	Validation Loss: 0.781628
Validation loss decreased (0.787273 --> 0.781628).  Saving model ...

Test Loss: 0.786072

Test Accuracy of airplane: 78% (787/1000)
Test Accuracy of automobile: 84% (844/1000)
Test Accuracy of  bird: 68% (683/1000)
Test Accuracy of   cat: 53% (534/1000)
Test Accuracy of  deer: 57% (575/1000)
Test Accuracy of   dog: 68% (689/1000)
Test Accuracy of  frog: 80% (805/1000)
Test Accuracy of horse: 75% (758/1000)
Test Accuracy of  ship: 84% (840/1000)
Test Accuracy of truck: 80% (809/1000)

Test Accuracy (Overall): 73% (7324/10000)