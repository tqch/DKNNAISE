# DKNNAISE
The platform for DKNN+AISE.
## Platform
* Python: 3.7
* PyTorch: 1.5.0
## Functions
1. Training CNNs and robust CNNs
2. Training DkNN
3. Obtain features (CNN representations) of each layer from any input
4. Calculating adversarial accuracy of CNNs and DkNN (under different attack confidence)
5. Adversarial attack (PGD, support target attack and non-target attack)
6. More functions will be released
## Howto
0. Change the device (current one is 'cuda:5') to fit your own computing resources. The CIFAR-10 dataset can be downloaded from its webpage or [here](https://github.com/wangren09/TrojanNetDetector/tree/master/DFTND/cifar10). The MNIST dataset can be downloaded from website or using the build-in function.
1. Train a CNN using MNIST_train.ipynb and cifar_train.ipynb (results include clean accuracy and adversarial accuracy. We already provide a trained model in the path /models/)
2. Run the file main_mnist.ipynb or main.ipynb.
3. The results will show the clean accuracy, adversarial accuracy, confidence, credibility.
4. Run Adv_Attack_MNIST.ipynb. The results will give you adversarial examples as well as perturbations and attack confidence.
