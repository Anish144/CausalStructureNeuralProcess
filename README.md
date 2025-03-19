# A Meta-Learning Approach to Bayesian Causal Discovery

This repository is the official implementation of [A Meta-Learning Approach to Bayesian Causal Discovery](https://arxiv.org/abs/2412.16577).

We use a transformer based meta-learning approach to directly approximate the Bayesian posterior over causal structures.

Inferring causal structure from observational data is a difficult task, namely due to identifiability issues and
finite sample effects. As such, it is important to be able to quantify uncertainty over causal structure to
facilitate downstream data collection (such as through active learning) that may increase the confidence over a single causal structure.

The main issue with computing the posterior over causal structures is two fold:
1. It requires inference over the causal mechanism, which can be intractable
2. The number of causal structures increases super-exponentially with the number of variables

We tackle the above issues by using a transformer neural process to directly learn the posterior over
casual strucure. It implicitly marginalises the causal mechanism without explicit calculation, and
handles the high dimensional causal space better than other causal structure learning methods.


>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -e .
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z.

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

