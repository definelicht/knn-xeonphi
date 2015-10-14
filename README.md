Links
-------------
- [kNN-CUDA](https://github.com/vincentfpgarcia/kNN-CUDA) - GPU implementation.
- [Handwritten digits](http://yann.lecun.com/exdb/mnist/) - Data set to test KNN implementation on.
- [FLANN](http://www.cs.ubc.ca/research/flann/) - Fast nearest neighbor search library.
- [ANN](https://www.cs.umd.edu/~mount/ANN/) - Approximate nearest neighbor search.
- [80 Million tiny images](http://people.csail.mit.edu/torralba/publications/80millionImages.pdf) - Large dataset used with NN algorithsm (760GB)
- [CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html) - Smaller 32x32 image based dataset

Things to try
-------------
- KNN: Distance computations using a matrix library, since this is essentially matrix operations and can be trivially sped up.
- Vary parameters deciding how many comparisons per level vs. depth.
