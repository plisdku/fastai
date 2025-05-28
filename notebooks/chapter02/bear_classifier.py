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
# # Bear classifier
#
# Here's my plan.
#
# 1. I will load the images into `bear_photos/grizzly` and so on
# 2. I will download a few hundred images of each bear type
# 3. I will try to train a Resnet-18 and maybe Resnet-34
# 4. I will do some data cleaning like in the book—sort my dataset by loss and use the filter widget
# 5. I'll make an upload widget to upload one or several images and predict what they are

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
from io import BytesIO
from pathlib import Path
from timeit import default_timer

import ipywidgets as widgets
from duckduckgo_search import DDGS
from fastai.data.block import CategoryBlock, DataBlock
from fastai.data.transforms import RandomSplitter, parent_label
from fastai.metrics import error_rate
from fastai.vision.all import (
    ClassificationInterpretation,
    ImageBlock,
    PILImage,
    download_images,
    get_image_files,
    load_learner,
    resize_images,
    verify_images,
    vision_learner,
)
from fastai.vision.augment import RandomResizedCrop, Resize
from fastai.vision.widgets import ImageClassifierCleaner

# %%
from IPython.display import clear_output
from torchvision.models import resnet18

# %%
queries = [
    "grizzly bear photo",
    "black bear photo",
    "koala bear photo",
    "teddy bear photo",
    "water bear photo",
]
max_results = 400

root = Path("bearpics")

for query in queries:
    term = query.split()[0]
    pic_dir = root / term
    if os.path.exists(pic_dir):
        continue

    print("Query:", query)
    ddgs = DDGS()
    urls = [result["image"] for result in ddgs.images(query, max_results=max_results)]

    pic_dir.mkdir(parents=True, exist_ok=True)

    download_images(pic_dir, urls=urls)
    clear_output(wait=True)

print("Resizing...")
tic = default_timer()
resize_images(root, recurse=True, max_size=400, dest=root)
toc = default_timer()
print("Duration:", toc - tic, "s")

# %%
failed = verify_images(get_image_files(root))
failed.map(Path.unlink)

# %%
# images = get_image_files(root)
# Image.open(images[0])

# %%
db = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(seed=42),
    get_y=parent_label,
    item_tfms=Resize(128, method="squish"),
)
# Replace the transforms.
db = db.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))

# Now

# %%
data_loaders = db.dataloaders(root, bs=32)

# %%
# data_loaders.show_batch(max_n=12)

# %%
model_fname = os.path.abspath("bear-model.pkl")
load_from_disk = os.path.exists(model_fname)
if load_from_disk:
    print("Load")
    learner = load_learner(model_fname)
else:
    learner = vision_learner(data_loaders, resnet18, metrics=error_rate)
    learner.fine_tune(4)
    learner.export(model_fname)

# %%
cl_interp = ClassificationInterpretation.from_learner(learner)
cl_interp.plot_confusion_matrix()

# %%
cl_interp.plot_top_losses(5, nrows=1)

# %%
cleaner = ImageClassifierCleaner(learner, max_n=50)
cleaner

# %%
cleaner.delete()

# %%
for idx in cleaner.delete():
    cleaner.fns[idx].unlink()

# %% [markdown]
# # Inference widget

# %%
upload_widget = widgets.FileUpload()
upload_widget

# %%
upload_widget.value

# %%
images = [PILImage.create(BytesIO(item["content"])) for item in upload_widget.value]

# %%
predictions = [learner.predict(item) for item in images]

# %%
m = learner.model

# %%

# %%
list(m.get_submodule("0.1").parameters())

# %%
m
