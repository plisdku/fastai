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
import os
from io import BytesIO

import ipywidgets as widgets
from fastai.vision.all import (
    ImageDataLoaders,
    PILImage,
    Resize,
    URLs,
    error_rate,
    get_image_files,
    load_learner,
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
model_fname = os.path.abspath("pet-model.pkl")
load_from_disk = os.path.exists(model_fname)
if load_from_disk:
    learn = load_learner(model_fname)
else:
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(1)
    learn.export(model_fname)

# %%
uploader = widgets.FileUpload()
uploader

# %%
img = PILImage.create(BytesIO(uploader.value[0]["content"]))
is_cat, _, probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
