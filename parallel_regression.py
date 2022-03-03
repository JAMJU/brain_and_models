#!/usr/bin/env python
# -*- coding: utf-8 -*-

from regressors import (
    build_regressors,
)
# Sklearn
from sklearn.preprocessing import RobustScaler
from utils import scale, correlate
from regressor_hier import HierarchRegressor
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from path_to_have import (
    stimuli_path_wav,
    path_to_english_data,
    path_to_french_data,
    path_to_mandarin_data,
    path_wav2vec_english,
    )

from multiprocessing import Pool
from preproc_stim import create_df_stimulus
from subject_data import Subjects
import os


def fitting(train_test, features_all_tasks, fmri_data_all_tasks, regressor):
    """
    Feating function
    :param train_test: training and test set, along with hierarch parameters, number of dimension and subject name
    :param features_all_tasks: features used for the regression
    :param fmri_data_all_tasks: target fmri
    :param regressor: parameters of the regressor
    :return:
    """
    print('Fitting')
    train = train_test[0]
    test = train_test[1]
    estimator = regressor
    hierarch = train_test[2]
    n_r = train_test[3]
    name = train_test[4]


    R = np.zeros((n_r, fmri_data_all_tasks.shape[1]))
    coeffs_reg = []
    train = [int(k) for k in train]
    test = [int(k) for k in test]
    estimator.fit(features_all_tasks[train], fmri_data_all_tasks[train])
    if not hierarch:
        fmri_predict = estimator.predict(features_all_tasks[test])
        R += correlate(fmri_data_all_tasks[test], fmri_predict)
    else:
        R += estimator.score(features_all_tasks[test], fmri_data_all_tasks[test])
        coeffs_reg = estimator.get_coefficient_importance()
    np.save( name + '.npy', R)
    return R, coeffs_reg


def get_feat_fmri(task_all_data, subjects):
    """
    Function to get fmri data from one subject and corresponding model's features
    :param task_all_data:  run number, subject_name, TR size, features wanted, model wanted, scaling, if random wanted, scaling features
    :param subjects: subject object
    :return:
    """

    # fmri data
    run_nb = task_all_data[0]
    subject = task_all_data[1]
    TR = task_all_data[2]

    # features data
    sel_features = task_all_data[3]
    model = task_all_data[4]
    scale_before = task_all_data[5]
    random_input = task_all_data[6]
    scale_features = task_all_data[7]


    print('loading fmri data')
    fmri = subjects(subject_id = subject, run_nb = run_nb)
    print(fmri.shape)

    # fmri prep
    shape_fmri = fmri.shape  # we save the shape

    # we can remove the first onsets
    onset_task = 0
    fmri_data_all = fmri[onset_task:, :]  # everything starts from 0

    print(run_nb)
    # we add the pulses in the stimuli dataframe
    print("adding pulses")
    n_pulse = len(fmri_data_all)
    events = create_df_stimulus(n_pulse=n_pulse, TR = TR)
    events = events.reset_index()


    assert len(fmri_data_all) == len(events.query("condition=='Pulse'"))
    assert int(
        events.query("condition=='Pulse'")["volume"].max()) == (len(fmri_data_all) - 1)

    # getting valid and unvalid voxels
    print('getting valid voxels')
    valid = fmri_data_all.std(axis=0) > 0
    nonvalid = fmri_data_all.std(axis=0) <= 0


    # Now we get features !!
    # Choose feature path
    # path of features or audio
    language_part = subjects.language
    path_to_data = ''
    if language_part == 'EN':
        path_to_data = path_to_english_data
    elif language_part == 'FR':
        path_to_data = path_to_french_data
    elif language_part == 'CN':
        path_to_data = path_to_mandarin_data

    stimuli_path_list = []
    for sel in sel_features:
        stimuli_path = ""
        if sel == "melfilterbanks" or sel == "mfccs" or sel == "rms":
            stimuli_path = stimuli_path_wav
        elif "wav2vec_english_" in sel:
            stimuli_path = os.path.join(path_to_data, path_wav2vec_english)
        else:
            print('NOT AVAILABLE')
        stimuli_path_list.append(stimuli_path)

    print('getting features')
    # Build regressors
    features, groups = build_regressors(
        events.copy(),
        model=model,
        sel_features=sel_features,
        stimuli_path_list=stimuli_path_list,
        shuffle=False,
        scale=scale_before,
        shuffle_files=False,
        random_input=random_input,
        task_wanted=str(run_nb),
        language_part=language_part
    )
    # print(groups)

    print("scaling features")
    if scale_features:
        features = scale(features)

    # we normalize
    print('normalizing fmri')
    scaler = RobustScaler(quantile_range=(0.01, 0.99), unit_variance=True)
    fmri_data_all[:, valid] = scaler.fit_transform(fmri_data_all[:, valid])

    return features, fmri_data_all, groups, nonvalid


def main(
    subject,
    model="glover",
    estimator_="ridge",
    sel_features=None,
    hierarch=None,
    scale_features=True,
    scale_before=True,
    random_input=(False, 0),
    tempname = "",
    language_part = 'EN',
    using_pca_fmri = False
):
    """
    computes from the name of a subject the pearson correlation between the
    fmri activations predicted by sel_features and the real ones
    inputs:
    confounds: 'all' if you want to consider them all, otherwise a list of the confounds to consider
    detrend: Detrend locally fmri, bool
    high_pass: apply a high pass filter to the fmri, bool
    detrend_context: detrend fmri using the context, bool
    n_delays: number of delays for the FIR preprocessing, int
    model: model used for the preprocessing of the acoustic features, "fir" or "glover", str
    estimator: model used for the regression, "ridge" or "scaler_pca_ridge", str
    sel_features: list of the codenames of the features you want to use, list of str
    hierarch: dictionnary containing {subtract: .., concat: ..} indicating what
    to do if you want a hierachical model, None if you do not want one
    scale_features: if you want to scale the features after the preprocessing
    scale_before: type of scaling you want on the feature before the preprocessing,
    can be a list to precise what you want for each feature. 'minmax' True of False.
    random_input: tuple, random_input[0]: if you want to replace the features by random
    float between 0 and 1, random_input[1]: what dimension you want, if 0 the dimension of the feature you selected

    outputs:
    stc: the value of R for each voxel, morphed on an average brain
    """


    # TR fixed
    TR = 2.0

    # we get subject data
    subjects = Subjects(lang=language_part)

    # First we get fmri data and features
    # we create a list of arguments (to allow for parallelization if wanted, not applied here)
    arguments =[[i] for i in range(1,10)]
    for i in range(len(arguments)):
        # fmri data
        arguments[i].append(subject)
        arguments[i].append(TR)

        # feature data
        arguments[i].append(sel_features)
        arguments[i].append(model)
        arguments[i].append(scale_before)
        arguments[i].append(random_input)
        arguments[i].append(scale_features)


    # We get features and fmri data
    all_feat_fmri = []
    for i in range(len(arguments)):
        all_feat_fmri.append(get_feat_fmri(arguments[i], subjects=subjects))

    del arguments
    features_all_tasks = np.concatenate([featt[0] for featt in all_feat_fmri], axis = 0)
    fmri_data_all_tasks = np.concatenate([featt[1] for featt in all_feat_fmri], axis = 0)

    groups = all_feat_fmri[0][2]
    nonvalid_ = [featt[3] for featt in all_feat_fmri]

    print('groups', groups)
    nonvalid = nonvalid_[0]
    for i in range(1,len(nonvalid_)):
        nonvalid_ = np.logical_and(nonvalid, nonvalid_[i])

    print('shape all features', features_all_tasks.shape)
    print('shape all fmri', fmri_data_all_tasks.shape)

    # release memory
    del all_feat_fmri


    # Prepare linear modeling
    alphas = np.logspace(1, 8, 20)


    print("Fit")
    pca_pipeline = (
        True if "pca" in estimator_ else False,
        "first" if estimator_ == "pca_scaler_ridge" else "other",
    )
    n_r = len(sel_features) # nb of levels of hierarchy
    cv = KFold(10, shuffle=False)
    splitting = cv.split(features_all_tasks)


    nb_splits = cv.n_splits
    print('nb splits', nb_splits)
    arguments = [[train_,test_] for train_, test_ in splitting]
    for i in range(len(arguments)):
        arguments[i].append(hierarch)
        arguments[i].append(n_r)
        arguments[i].append(tempname + str(i))


    print('nb arg', len(arguments))


    for i in range(nb_splits):
        print(i, 'split')
        regressor = HierarchRegressor(model = RidgeCV(alphas, alpha_per_target=True), hierarchy = groups,pca_pipeline=pca_pipeline,
                                      concat = hierarch['concat'],subtract = hierarch['subtract'], using_pca_fmri=using_pca_fmri)
        R, coeff = fitting(arguments[i], features_all_tasks=features_all_tasks,
                              fmri_data_all_tasks=fmri_data_all_tasks, regressor=regressor)

        if i== 0:
            R_final = R
            coeff_final = coeff
        else:
            R_final += R
            coeff_final += coeff
        if i < nb_splits -1:
            del regressor

    del arguments

    R_final = R_final / float(nb_splits)


    # we deal with coefficients
    coeff_final = coeff_final /float(nb_splits) # we average over splits, coeffs is irm x feat
    coeff_final = np.absolute(coeff_final) # we take the absolute value
    value_per_layer = []
    for level in np.unique(regressor.hierarchy):
        columns = np.where(regressor.hierarchy == level)[0]
        coeff_layer = coeff_final[:,columns]
        coeff_layer = np.mean(coeff_layer, axis = 1) # fmri x 1
        value_per_layer.append(coeff_layer.copy())

    value_per_layer = np.asarray(value_per_layer)

    print('shape_result', R_final.shape)
    print('coeff shape results', value_per_layer.shape)
    # we make sure that the non valid stay at nan
    R_final[:, nonvalid] = np.nan
    value_per_layer[:,nonvalid] = np.nan
    print('done!')
    return R_final, value_per_layer
