# CSE 599i (Generative Models) Homework 1 #

In this part of Homework 1, your task it to implement a transformer, and run several experiments with it using the Wikitext-2 dataset.

The Wikitext-2 dataset is provided in this dataset (under the `data/` directory) and I've provided code for loading and processing this data into minibatches in `dataset.py`.

A scaffolding for the transformer logic is provided in `transformer.py`. You will need to implement the `forward` function in the `TransformerBlock` class (about 10 lines of code).

Framework code for training the model is found in `notebook.ipynb`. I recommend using Google Colab to execute this notebook (with a GPU accelerator attached). If you choose to use your own setup, you will need to delete the first cell, and possibly modify the paths in the second cell to reflect your local setup.
