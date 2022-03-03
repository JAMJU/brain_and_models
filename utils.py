#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def correlate(X, Y):
    """ Compute error score"""
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    out = np.zeros(max([Y.shape[1], X.shape[1]]))
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    X = np.array(X, dtype = float)
    Y = np.array(Y, dtype=float)

    SX2 = (X ** 2).sum(0) ** 0.5
    SY2 = (Y ** 2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    valid = (SX2 != 0) & (SY2 != 0)
    out[valid] = SXY[valid] / (SX2[valid] * SY2[valid])
    return out


def scale(X):
    """ Safe rescaling """
    X = np.asarray(X)
    shape = X.shape
    if len(shape) == 1:
        X = X[:, None]
    X = X - np.nanmean(X, 0, keepdims=True)
    std = np.nanstd(X, 0, keepdims=True)
    non_zero = np.where(std > 0)
    X[:, non_zero] /= std[non_zero]
    X[np.isnan(X)] = 0
    X[~np.isfinite(X)] = 0

    return X.reshape(shape)
