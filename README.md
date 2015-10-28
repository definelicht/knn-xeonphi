Links
-----
- [kNN-CUDA](https://github.com/vincentfpgarcia/kNN-CUDA) - GPU implementation.
- [FLANN](http://www.cs.ubc.ca/research/flann/) - Fast nearest neighbor search library.
- [ANN](https://www.cs.umd.edu/~mount/ANN/) - Approximate nearest neighbor search.
- [kNN Standard](http://web.stanford.edu/class/cs106l/handouts/assignment-3-kdtree.pdf) - kd-trees and KNN explained in simple English.

Data Sets
---------
- [Handwritten digits](http://yann.lecun.com/exdb/mnist/) - Data set to test KNN implementation on.
- [80 million tiny images](http://people.csail.mit.edu/torralba/publications/80millionImages.pdf) - Large dataset used with NN algorithms (760GB).
- [CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html) - Smaller 32x32 image based dataset
- [TexMex](http://corpus-texmex.irisa.fr/) - Datasets for approximate nearest neighbor search (1 billion vectors!)

Things to try
-------------
- Distance computations using a matrix library, since this is essentially matrix operations and can be trivially sped up.
- Vary parameters deciding how many comparisons per level vs. depth.

Influences on performance from data
-----------------------------------
- Dimensionality, but can also be positive: high dimensionality might cluster better etc. (described in FLANN).
- Correlation, same reason(s) as above.

Influences on performance from application
------------------------------------------
- Cost of searching tree
- Cost of constructing tree
- Memory requirement
