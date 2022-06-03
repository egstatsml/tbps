# TBPS
A package for performing inference using the Stochastic Bouncy Particle Sampler (SBPS).

In particular, it implements methods for the Adaptive Thinning SBPS (atSBPS) and the efficient SBPS (eSBPS).


This module is built upon the Tensorflow Probability library to enable GPU acceleration, and to make modelling more amenable to existing inference methods built within.

An example to run a model would,


``` shell
python train_conv.py resnet20 categorical 0.1 \
       --out_dir <out_directory_path> \
       --batch_size 256 \
       --data cifar_10 \
       --num_results 2000 \
       --num_burnin 0 \
       --bps \
       --ipp_sampler adaptive \
       --map_path <path_to_map_weights> \
```
