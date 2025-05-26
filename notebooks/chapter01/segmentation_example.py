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
import numpy as np
from fastai.vision.all import (
    SegmentationDataLoaders,
    URLs,
    get_image_files,
    unet_learner,
    untar_data,
)
from torchvision.models import resnet34

# %%
path = untar_data(URLs.CAMVID_TINY)

# %%
fnames = get_image_files(path / "images")
codes = np.loadtxt(path / "codes.txt", dtype=str)

# %%
label_func = lambda o: path / "labels" / f"{o.stem}_P{o.suffix}"  # noqa
dls = SegmentationDataLoaders.from_label_func(path, bs=8, fnames=fnames, label_func=label_func, codes=codes)

# %%
learn = unet_learner(dls, resnet34)

# %%
learn.fine_tune(8)

# %%
learn.show_results(max_n=6, figsize=(7, 18))

# %%
