"""
Copyright (c) 2021 Bahareh Tolooshams

train for the model x = Dz

:author: Bahareh Tolooshams
"""

import numpy as np
import scipy as sp
import torch
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.gridspec as gridspec


def visualize_dense_dictionary(D, save_path, reshape=(28, 28), cmap="gray"):
    p = D.shape[-1]
    a = np.int(np.ceil(np.sqrt(p)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    for col in range(p):
        ax1 = plt.subplot(gs1[col])
        wi = D[:, col].clone().detach().cpu().numpy()
        if reshape:
            wi = np.reshape(wi, reshape)
        plt.imshow(wi, cmap=cmap)
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_conv_dictionary(D, save_path, cmap="gray"):
    p = D.shape[0]
    a = np.int(np.ceil(np.sqrt(p)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    W = D.clone().detach().cpu().numpy()
    W = (W - np.min(W)) / (np.max(W) - np.min(W))
    for col in range(p):
        ax1 = plt.subplot(gs1[col])
        wi = W[col]
        wi = np.transpose(wi, (1, 2, 0))
        plt.imshow(wi, cmap=cmap)
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_conv_feature_maps(Z, save_path, cmap="afmhot"):
    p = Z.shape[0]
    a = np.int(np.ceil(np.sqrt(p)))
    fig = plt.figure(figsize=(a, a))
    gs1 = gridspec.GridSpec(a, a)
    gs1.update(wspace=0.025, hspace=0.05)
    for col in range(p):
        ax1 = plt.subplot(gs1[col])
        wi = Z[col].clone().detach().cpu().numpy()
        plt.imshow(wi, cmap=cmap)
        plt.axis("off")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect("equal")
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def visualize_image(x, xhat, save_path, cmap="gray"):

    ################################
    axes_fontsize = 10
    legend_fontsize = 10
    tick_fontsize = 10
    title_fontsize = 10

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
        }
    )

    fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)

    for ax in axn.flat:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(x, (1, 2, 0)), cmap=cmap)
    plt.title("img")

    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(xhat, (1, 2, 0)), cmap=cmap)
    plt.title("est")

    fig.tight_layout(pad=0.0, w_pad=0.1, h_pad=0.1)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()
