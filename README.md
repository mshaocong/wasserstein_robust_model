# Minimal Implementation of Wasserstein Robust Model
This repo includes the minimal implementation of the classical **Wasserstein robust model** which is proposed in [[1]](#1). It generates the adversarial attacking data to improve the robustness of training model. Given the data set `(x, y)` where `x` is the image and `y` is the label, The goal of Wasserstein robust model is to solve the following optimization problem:

![Loss](/img/goal.png)

Here `z` denotes the adversarial data and `theta` denotes the parameters of deep neural network. `gamma` is used to assure the maximization problem to be strongly concave; it requires to be tuned for a different dataset. We expect the output parameter `theta` is the best model on all data similar to the original data rather than solely on the original data.  



We notice that all existing implementations on GitHub lack of the flexibility when we need 

* to customize the optimizer and the adversarial loss;

* to apply more than one step gradient ascent updates to solve the maximization problem; 

* to utilize the training history to boost up the speed of training.

  

## Framework

This repo includes the following parts:

* A highly customizable PyTorch adversarial training pipeline.  
* A repeatable experiment on MNIST dataset.

We illustrate the training pipeline as follows. 



For the first step, we do one step stochastic gradient descent (or other optimization method) on the neural network parameter by minimizing the *classifier loss*.

![Step1](/img/step1.png)



Since we have already evaluated the *classifier loss*, in the second step, it suffices to evaluate the *adversarial loss* and apply the stochastic gradient ascent. 

![Step2](/img/step2.png)



## Experiment







## References

<a id="1">[1]</a> Sinha, Aman, et al. "Certifying some distributional robustness with principled adversarial training." *arXiv preprint arXiv:1710.10571* (2017).



