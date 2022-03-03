#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from nistats.hemodynamic_models import compute_regressor
from features import get_features



def build_feature_data(
    sel,
    stimuli_path,
    scale=False,
    shuffle=False,
    shuffle_files=False,
    random_input=(False, 0),
    task_wanted = '',
    language_part = 'EN'):

    if task_wanted == '':
        raise ValueError

    if isinstance(scale, (str, bool)):
        scale = [scale]
    if isinstance(shuffle, (str, bool)):
        shuffle = [shuffle]

    # Define potential causal factors
    feature_array = list()
    durations_values = list()  # for glover model

    onsets, features, durations = get_features(
        stimuli_path=stimuli_path,
        feature_set=sel,
        shuffle_files=shuffle_files,
        random_input=random_input,
        task_wanted = task_wanted,
        language_part = language_part
    )

    eps = 10**(-10)

    if scale[0] == "minmax":
        features -= features.min(0) # starts to 0
        a = features.max(0)
        a[a == 0.] = 1.
        features /= a

    elif scale[0] is True:
        features -= features.mean(0)
        features /= features.std(0)

    if shuffle[0]:
        jdx = np.random.permutation(len(features))
        features_perm = features[jdx]
        features = features_perm

    features = np.nan_to_num(features)
    feature_array.extend(
        [features[i] for i in range(len(features))]
    )  # each feature can have different time x dim

    durations_values.extend(durations.tolist())
    return (
        feature_array,
        onsets,
        np.asarray(durations_values),
    )








def build_regressors(
    events,
    model="glover",
    sel_features=None,
    stimuli_path_list=list(),
    scale=True,
    shuffle=False,
    shuffle_files=False,
    random_input=(False, 0),
    task_wanted = '',
    language_part = 'EN'
):

    # Get fMRI frame times
    frame_times = events.query('type=="Pulse"').onset

    #print(model, len(sel_features))


    if model == "glover":
        # in this case we have more types
        reg_signals = list()
        groups_ = list()

        for i in range(len(sel_features)):
            features, onsets, durations = build_feature_data(
                sel=sel_features[i],
                stimuli_path=stimuli_path_list[i],
                scale=scale[i],
                shuffle=shuffle,
                random_input=random_input,
                shuffle_files=shuffle_files,
                task_wanted=task_wanted,
                language_part=language_part
            )
            features = np.asarray(features)
            reg_signals_temp = list()
            groups_temp = list()

            for values in features.T:

                if len(values.shape) == 1 or values.shape[1] == 1:
                    values = values[:, None]

                for column, v in enumerate(values.T):
                    #print('frames',frame_times)
                    frame_times = np.asarray(list(frame_times))
                    #print(onsets, 'onsets')
                    signal, name_ = compute_regressor(
                        np.c_[onsets, durations, v].T,
                        # initially ones instead of durations probably need to adapt that if multiple features set
                        hrf_model=model,
                        frame_times=frame_times,
                        oversampling=16,
                    )

                    reg_signals_temp.append(signal)

                    groups_temp.extend([i] * signal.shape[1])
            groups_.extend(groups_temp)
            reg_signals.extend(reg_signals_temp)

        reg_signals = np.concatenate(reg_signals, 1)
        groups = np.asarray(groups_)
    else:
        raise ValueError


    print('groups created', groups)
    return reg_signals, groups


