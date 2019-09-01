# Class-balanced-loss-pytorch
Pytorch implementation of the paper
[Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555) presented at CVPR'19.


[Yin Cui](https://ycui.me/), Menglin Jia, [Tsung-Yi Lin](https://vision.cornell.edu/se3/people/tsung-yi-lin/)(Google Brain), [Yang Song](https://ai.google/research/people/author38270)(Google), [Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/)

## Dependencies
- Python (>=3.6)
- Pytorch (>=1.2.0)

## How it works

It works on the principle of calculating effective number of samples for all classes which is defined as:

![alt-text](https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/samples.png)

Thus, the loss function is defined as:

![alt-text](https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/loss.png)

Visualisation for effective number of samples


![alt-text](https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/image.png "Visualisation for effective number of samples")

## References

[official tensorflow implementation](https://github.com/richardaecn/class-balanced-loss)
