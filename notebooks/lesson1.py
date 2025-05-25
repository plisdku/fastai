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
# %load_ext autoreload
# %autoreload 2

# %%
import os
from pathlib import Path

from duckduckgo_search import DDGS
from fastai.data.block import CategoryBlock, DataBlock
from fastai.data.transforms import RandomSplitter, parent_label
from fastai.metrics import error_rate
from fastai.vision.augment import Resize
from fastai.vision.core import PILImage
from fastai.vision.data import ImageBlock
from fastai.vision.learner import vision_learner
from fastai.vision.utils import download_images, get_image_files, resize_images, verify_images
from fastcore.all import L
from fastdownload import download_url
from PIL import Image
from torchvision.models import resnet18

# %% [markdown]
# ## My homework
#
# Train a classifier.

# %%
searches = "forest", "bird"
max_results = 400

root = Path("results")

for term in searches:
    result_dir = root / term
    os.makedirs(result_dir, exist_ok=True)

    ddgs = DDGS()
    img_urls = L(ddgs.images(term, max_results=max_results)).itemgot("image")
    download_images(result_dir, urls=img_urls)
    resize_images(result_dir, max_size=400, dest=result_dir)

# %%
failed = verify_images(get_image_files(root))
failed.map(Path.unlink)

# %%
data_block = DataBlock(
    blocks=[ImageBlock, CategoryBlock],  # input image, output category
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method="squish")],
)
data_loaders = data_block.dataloaders(root, bs=32)

# %%
data_loaders.show_batch(max_n=6)

# %%
learn = vision_learner(data_loaders, resnet18, metrics=error_rate)
learn.fine_tune(3)

# %%
# Get some images to test on

ddgs = DDGS()
test_url = L(ddgs.images("parrot", safesearch="off", max_results=1)).itemgot("image")
download_url(test_url[0], "test.jpg", show_progress=False)

# %%
test_img = Image.open("test.jpg").to_thumb(256, 256)

# %%
test_img

# %%
is_bird, _, probs = learn.predict(PILImage.create(test_img))
print(f"This is a {is_bird}")
print(f"Probability {probs[0]:0.4f}")

# %%
