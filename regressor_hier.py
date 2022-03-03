#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import correlate
from sklearn.base import TransformerMixin, clone
import numpy as np

from sklearn.decomposition import PCA


class HierarchRegressor(TransformerMixin):
    """Regress out X_n prediction on Y, to see whether X_n+1 still accounts
    for variance in Y - (Y_pred_n)
    """

    def __init__(
        self,
        model,
        hierarchy,
        pca_pipeline=(False, "other"),
        concat=True,
        subtract=True,
        using_pca_fmri = False,
    ):
        self.model = model
        self.hierarchy = hierarchy
        #print("hier", self.hierarchy)
        self.concat = concat
        self.subtract = subtract
        self.pca_pipeline = pca_pipeline
        self.using_pca_fmri = using_pca_fmri
        #print(self.subtract, self.concat, self.pca_pipeline)

    def _get_level(self, level):
        if self.concat:
            columns = np.where(self.hierarchy <= level)[0]
        else:
            columns = np.where(self.hierarchy == level)[0]
        return columns

    def fit(self, X, Y_):
        assert X.shape[1] == len(self.hierarchy)

        # we compute pca to reduce fmri dimensionality

        if self.using_pca_fmri:
            print('computing pca')
            self.pca_fmri = PCA(n_components=100)
            Y = self.pca_fmri.fit_transform(Y_)
            print(Y.shape)
        else:
            Y = Y_

        self.models_ = list()
        for level in np.unique(self.hierarchy):
            columns = self._get_level(level)
            model = clone(self.model)
            # if pca in pipeline, we adapt the nb of dimension (if feat dim less than the dimension chosen)
            if self.pca_pipeline[0]:
                # the pca is the first step or the second depending on option
                place = 0 if self.pca_pipeline[1] == "first" else 1
                n_c = model[place].n_components
                if n_c > len(
                    columns
                ):  # we check the dimension are enough for the pca to be applied
                    model[place].n_components = len(columns)
                #print(len(columns), model[place].n_components)
            # fit
            model.fit(X[:, columns], Y)
            # store
            self.models_.append(model)
            # residuals
            if self.subtract:
                Y = Y - model.predict(X[:, columns])

        return self

    def score(self, X, Y):
        assert X.shape[1] == len(self.hierarchy)
        R = np.zeros([len(self.models_), *Y.shape[1:]])
        for level, model in enumerate(self.models_):
            columns = self._get_level(level)
            Y_pred = model.predict(X[:, columns])
            print('Y pred', Y_pred.shape)

            if self.using_pca_fmri:
                Y_pred = self.pca_fmri.inverse_transform(Y_pred)
            # score
            R[level] = correlate(Y, Y_pred)
            # residual
            if self.subtract:
                Y = Y - Y_pred

        return R

    def get_coefficient_importance(self):
        """ Get coefficients of last concatenation """
        coeffs = self.pca_fmri.inverse_transform(np.swapaxes(self.models_[-1].coef_, 0, 1)) # original coef ar of size :size n_fmri_pca (100) x n_sound_features, we invert that
        # coeffs is now sound feat x irm
        # we invert that
        coeffs = np.swapaxes(coeffs, 0, 1) # irm x feat
        return coeffs



