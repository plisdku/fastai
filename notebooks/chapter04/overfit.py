# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# %xmode plain

# %%
import holoviews as hv
import ipywidgets as widgets
import torch
from IPython.display import display
from pch_math import polyval

hv.extension("bokeh")  # type: ignore

# %% [markdown]
# Here's the idea. Make some data, $(x_i, y_i)$ and try to fit it with various models using autograd.

# %%
xs = torch.linspace(-1, 1, 11)
ys = xs**2 + 0.2 * torch.randn(xs.shape)

curve1 = hv.Curve((xs, ys), label="blah").opts(line_dash="solid")
curve2 = hv.Points((xs, ys + 0.2 * torch.randn(xs.shape)), label="blee").opts(marker="o", color="r", size=8)
(curve1 * curve2).opts(width=600, height=500)

# plt.plot(xs, ys, 'o-')

# %%

# %%
polyval([1, 0, 0], 4)

# %%


yyy = polyval(torch.tensor([1, 1]), xs)
(hv.Curve((xs, yyy)) * hv.Scatter((xs, yyy)).opts(size=4)).opts(width=1000, height=700)

# %%
alpha = 0.1
num_iters = 100

xs = torch.linspace(-1, 1, 101)

params = torch.tensor((1, -0.1, 0.1)).clone().detach()
params.requires_grad = True


def model(params):
    return polyval(params, xs)


def widget_plot(a, b, c):
    p_truth = torch.tensor((a, b, c))
    ys = model(p_truth) + 0.25 * torch.randn(xs.shape)

    def mse(params):
        return torch.mean((model(params) - ys) ** 2)

    curves = []

    params = p_truth.clone()
    params.requires_grad = True

    for nn in range(num_iters):
        y = model(params.detach())

        if nn % 4 == 0:
            curves.append(hv.Curve((xs, y)))

        mse(params).backward()
        with torch.no_grad():
            params -= alpha * params.grad
            params.grad.zero_()

    curves.append(hv.Scatter((xs, ys)).opts(size=4))
    display(hv.Overlay(curves).opts(width=800, height=300))


widgets.interact(
    widget_plot,
    a=widgets.FloatSlider(value=0, min=-1, max=1, step=0.1),
    b=widgets.FloatSlider(value=0, min=-1, max=1, step=0.1),
    c=widgets.FloatSlider(value=0, min=-1, max=1, step=0.1),
    continuous_update=False,
)

# %%
