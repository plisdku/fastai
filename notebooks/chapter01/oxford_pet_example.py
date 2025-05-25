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

# %%
from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    URLs,
    error_rate,
    get_image_files,
    resnet34,
    untar_data,
    vision_learner,
)

# %%
path = untar_data(URLs.PETS) / "images"


def is_cat(x):
    return x[0].isupper()


dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42, label_func=is_cat, item_tfms=Resize(224)
)

# %%
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

# %%
