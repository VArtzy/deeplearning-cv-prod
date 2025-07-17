# Typically ML powered app components:

## Software 2.0 case: OCR Text Recognizer

### Frontend and Backend
What you see above is the "frontend", the user-facing component, of the application.

Frontend web development is typically done using Javascript as the programming language. Most ML is done in Python (see below), so we will instead build our frontend using the Python library Gradio.

Another excellent choice for pure Python web development might be Streamlit or even, in the near future, tools built around PyScript.

Notice the option to "flag" the model's outputs. This user feedback will be sent to Gantry, where we can monitor model performance, generate alerts, and do exploratory data analysis.

The model that reads the image to produce the text is not running in the same place as the frontend. The model is the "backend" of our application. We separate the two via a JSON API. The model is deployed serverlessly to Amazon Web Services using AWS Lambda, which runs a Docker container that wraps up our model.

Docker is the tool of choice for virtualization/containerization. As containerized applications become more complex, container orchestration becomes important. The premier tool for orchestrating Docker containers is kubernetes, aka k8s. Non-experts on cloud infrastructure will want to use their providers' managed service for k8s, e.g. AWS EKS or Google Kubernetes Engine.

The container image lives inside the Elastic Container Registry, a sort of "GitHub for Docker" on AWS. The choice to go serverless makes it effortless to scale up our model across a number of orders of magnitude and the choice to containerize reduces friction and error when moving our model from development to production.

This could equally as well be done on another cloud, like Google Cloud Platform or Microsoft Azure, which offer serverless deployment via Google Cloud Functions and Azure Functions, respectively.

The backend operates completely independently of the frontend, which means it can be used in multiple contexts.

### Model Training
Let's start back at the beginning -- developing a model. We'll then make our way back to where we left off above, the handoff from model development/training to the actual application.

We begin by training a neural network (a ResNet encoder to process the images and a Transformer decoder to produce the output text).

Neural networks operate by applying sequences of large matrix multiplications and other array operations. These operations are much faster on GPUs than on CPUs and are relatively easy to parallelize across GPUs. This is especially true during training, where many inputs are processed in parallel, or "batched" together.

Purchasing GPUs and properly setting up a multi-GPU machine is challenging and has high up-front costs. So we run our training via a cloud provider, specifically Lambda Labs GPU Cloud.

Other cloud providers offer GPU-accelerated compute but Lambda Labs offers it at the lowest prices, as of August 2022. Larger organizations may benefit from the extra features that integration with larger cloud providers, like AWS or GCP, can provide (e.g. unified authorization and control planes). Because independent, full-stack developers are often very price-sensitive, we recommend Lambda Labs -- even more, we recommend checking current and historical instance prices.

For smaller units of work, like debugging and quick experiments, we can use Google Colaboratory, which provides limited access to free GPU (and TPU) compute in an ephemeral environment.

For small-to-medium-sized deep learning tasks, Colab Pro (
50/mo.) can be competitive with the larger cloud providers.

If you're running this notebook on a machine with a GPU, e.g. on Colab, running the cell below will show some basic information on the GPU's state.

!nvidia-smi
Because the heavy work is done on the GPU, using lower-level libraries, we don't need to write the majority of our model development code in a performant language like C/C++ or Rust.

We can instead write in a more comfortable, but slower language: it doesn't make sense to drive an F1 car to the airport for an international flight.

The language of choice for deep learning is Python.

import this  # The Zen of Python
We don't want to write our Python library for GPU acceleration from scratch, especially because we also need automatic differentiation -- the ability to take derivatives of our neural networks. The Python/C++ library PyTorch offers GPU-accelerated array math with automatic differentiation, plus special neural network primitives and architectures.

There are two major alternatives to PyTorch for providing accelerated, differentiable array math, both from Google: early mover TensorFlow and new(ish)comer JAX. The former is more common in certain larger, older enterprise settings and the latter is more common in certain bleeding-edge research settings. We choose PyTorch to split the difference, but can confidently recommend all three.

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"  # run on GPU if available

# create an array/tensor and track its gradients during calculations
a = torch.tensor([1.], requires_grad=True) \
  .to(device)  # store the array data on GPU (if available)
b = torch.tensor([2.]).to(device)

# calculate new values, building up a "compute graph"
c = a * b + a

# compute gradient of c with respect to a by "tracing the graph backwards"
g, = torch.autograd.grad(outputs=c, inputs=a)

g
PyTorch provides a number of features required for creating deep neural networks, but it doesn't include a high-level framework for training or any of a number of related engineering tasks, like metric calculation or model checkpointing.

We use the PyTorch Lightning library as our high-level training engineering framework.

PyTorch Lightning is the framework of choice for generic deep learning in PyTorch, but in natural language processing, many people instead choose libraries from Hugging Face. Keras is the framework of choice for TensorFlow. In some ways, Flax is the same for JAX; in others, there is not as of July 2022 a high-level training framework in JAX.

### Experiment and Artifact Tracking
ML models are challenging to debug: their inputs and outputs are often easy for humans to interpret but hard for traditional software programs to understand.

They are also challenging to design: there are a number of knobs to twiddle and constants to set, like a finicky bunch of compiler flags. These are known as "hyperparameters".

So building an ML model often looks a bit less like engineering and a bit more like experimentation. These experiments need to be tracked, as do large binary files, or artifacts, that are produced during those experiments -- like model weights.

We choose Weights & Biases as our experiment and artifact tracking platform.

MLFlow is an open-source library that provides similar features to W&B, but the experiment and artifact tracking server must be self-hosted, which can be burdensome for the already beleaguered full-stack ML developer. Basic experiment tracking can also be done using Tensorboard, and shared using tensorboard.dev, but Tensorboard does not provide artifact tracking. Artifact tracking and versioning can be done using Git LFS, but storage and distribution via GitHub can be expensive and it does not provide experiment tracking. Hugging Face runs an alternative git server, Hugging Face Spaces, that can display Tensorboard experiments and mandates Git LFS for large files (where large means >10MB).

The resulting experiment logs can be made very rich and are invaluable for debugging (e.g. tracking bugs through the git history) and communicating results inside and across teams.

Logged data is inert. It becomes usable, actionable information when it is given context and form.

### The Handoff to Production
PyTorch Lightning produces large artifacts called "checkpoints" that can be used to restart model training when it stops or is interrupted (which allows the use of much cheaper "preemptible" cloud instances).

We store these artifacts on Weights & Biases.

When they are ready to be deployed to production, we compile these model checkpoints down to a dialect of Torch called torchscript that is more portable: it drops the training engineering code and produces an artifact that is executable in C++ or in Python. We stick with a Python environment for simplicity.

TensorFlow has similar facilities for delivering models, including tensorflow.js and TensorFlow Extended (TFX). There are also a number of alternative portable runtime environments for ML models, including ONNX RT.

These deployable models are also stored on Weights & Biases, which connects them to rich metadata, including the experiments and training runs that produced the checkpoints from which they were derived.

We can pull this file down, package it into a Docker container via a small Python script, and ship it off to a container registry, like AWS ECR or Docker Hub, so that it can be used to provide the backend to our application.

### Application Diagram, Redux

![application diagram](https://imgur.com/a/rIehw4m)
