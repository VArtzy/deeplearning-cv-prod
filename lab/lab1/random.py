# re-execute this cell for more samples
import random

import wandb  # just for some convenience methods that convert tensors to human-friendly datatypes

import text_recognizer.metadata.mnist as metadata # metadata module holds metadata separate from data

idx = random.randint(0, len(x_train))
example = x_train[idx]

print(y_train[idx])  # the label of the image
wandb.Image(example.reshape(*metadata.DIMS)).image  # the image itself
