# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: fastai
#     language: python
#     name: fastai
# ---

# %% [markdown]
# # Sentiment analysis
#
# I was unable to train this on my laptop with `bs=64` (default) and with `bs=16`.
# Both times ran out of available GPU memory.
#
# In a Kaggle notebook.

# %%
from fastai.text.all import (
    AWD_LSTM,
    TextDataLoaders,
    URLs,
    accuracy,
    text_classifier_learner,
    untar_data,
)

# %%
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid="test", bs=16)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)

# %%
