At its core, PyTorch is a library for

doing math on arrays
with automatic calculation of gradients
that is easy to accelerate with GPUs and distribute over nodes.
Much of the time, we work at a remove from the core features of PyTorch, using abstractions from torch.nn or from frameworks on top of PyTorch.

This tutorial builds those abstractions up from core PyTorch, showing how to go from basic iterated gradient computation and application to a solid training and validation loop. It is adapted from the PyTorch tutorial What is torch.nn really?.

We assume familiarity with the fundamentals of ML and DNNs here, like gradient-based optimization and statistical learning. For refreshing on those, we recommend 3Blue1Brown's videos or the NYU course on deep learning by Le Cun and Canziani
