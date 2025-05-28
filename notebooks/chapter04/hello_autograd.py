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
import matplotlib.pyplot as plt
import torch

# %% [markdown]
# I'm going to show how to get the gradient of $y = x^2$. This sort of surprised me when I did it first.
#
# We are not going to start with $y = x^2$ and differentiate it to get $y' = 2x$. No, that would be too easy.
# Autograd works on scalars.
#
# The idea is to define some value
# $$
# S = \sum_{i=0}^{n-1} x_i^2
# $$
# and _then_ when we take the gradient, we get
# $$
# \partial_i S = 2 x_i.
# $$
#

# %%
xx = torch.linspace(-1, 1, 101, requires_grad=True)


def squazmo(x):
    return x**2


y = squazmo(xx)
sum_y = y.sum()
sum_y.backward()

plt.plot(xx.detach(), y.detach())
plt.plot(xx.detach(), xx.grad, "--")

# %% [markdown]
# Now let's figure out how to get the gradient with respect to one variable but not another.

# %%
xx = torch.linspace(-1, 1, 101, requires_grad=True)
yy = torch.linspace(-1, 1, 101)

zz = (xx + yy).sum()
zz.backward()

assert xx.grad is not None
assert yy.grad is None

plt.plot(xx.detach(), xx.grad)
