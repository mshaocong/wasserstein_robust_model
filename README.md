# Minimal Implementation of Wasserstein Robust Model
This repo includes the minimal implementation of the classical **Wasserstein robust model** which is proposed in [[1]](#1). It generates the adversarial attacking data to improve the robustness of training model. The goal of such model is to solve the following optimization problem:



We notice that all existing implementations on GitHub lack of the flexibility when we need 

* to customize the optimizer, the proximal operator, and the regularization;
* to apply more than one step gradient ascent updates in the inner loop; 
* to utilize the training history to boost the speed of training.

  



## Components

This repo includes the following parts:

* A highly customizable Pytorch adversarial training pipeline.  
* A repeatable experiment on MNIST dataset.



## Experiment Setup







## References

<a id="1">[1]</a> Sinha, Aman, et al. "Certifying some distributional robustness with principled adversarial training." *arXiv preprint arXiv:1710.10571* (2017).



