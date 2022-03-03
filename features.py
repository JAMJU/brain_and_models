#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import random as rd
import librosa




def add_event_sound(stimuli_dir, type_, shuffle_files=False, task_wanted = '', language_part = 'EN'):
    """
    Add acoustic event to the global event array
    """
    print('computing', type_, 'for', task_wanted)
    assert os.path.isdir(stimuli_dir)

    filename = ''
    if language_part == 'EN':
        filename = 'wave_english_run' + str(task_wanted)
    elif language_part == 'FR':
        filename = 'wave_french_RMS_run' + str(task_wanted)
    elif language_part == 'CN':
        filename = 'task-lppCN_section_' + str(task_wanted)
    # First we extract the feature for the file
    if type_ == "melfilterbanks":
        fname = os.path.join(stimuli_dir, filename + '.wav')
        if shuffle_files:
            print("shuffling...")
            print(fname)
            fname = os.path.join(
                stimuli_dir, rd.choice(list(os.listdir(stimuli_dir)))
            )
            if "index" in fname:
                fname = os.path.join(
                    stimuli_dir, rd.choice(list(os.listdir(stimuli_dir)))
                )
            print(fname)
        y, sr = librosa.load(fname)
        time_hop = 0.010
        spect = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            win_length=int(0.025 * sr),
            hop_length=int(0.010 * sr),
        )
        spect = librosa.amplitude_to_db(spect)
        spect = spect.T
        dimension = spect.shape[1]
        duration = 0.025

    elif type_ == "mfccs":
        fname = os.path.join(stimuli_dir, filename + '.wav')
        if shuffle_files:
            print("shuffling...")
            print(fname)
            fname = os.path.join(
                stimuli_dir, rd.choice(list(os.listdir(stimuli_dir)))
            )
            if "index" in fname:
                fname = os.path.join(
                    stimuli_dir, rd.choice(list(os.listdir(stimuli_dir)))
                )
            print(fname)
        y, sr = librosa.load(fname)
        time_hop = 0.010
        spect = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            win_length=int(0.025 * sr),
            hop_length=int(0.010 * sr),
        )

        spect = spect.T
        dimension = spect.shape[1]
        duration = 0.025

    elif type_ == "rms":
        fname = os.path.join(stimuli_dir, filename + '.wav')
        if shuffle_files:
            print("shuffling...")
            print(fname)
            fname = os.path.join(
                stimuli_dir, rd.choice(list(os.listdir(stimuli_dir)))
            )
            if "index" in fname:
                fname = os.path.join(
                    stimuli_dir, rd.choice(list(os.listdir(stimuli_dir)))
                )
            print(fname)
        y, sr = librosa.load(fname)
        time_hop = 0.010
        spect = librosa.feature.rms(
            y=y, frame_length=int(0.025 * sr), hop_length=int(0.010 * sr)
        )
        spect = spect.T
        dimension = 1
        duration = 0.025


    elif ("wav2vec" in type_
    ):
        if 'conv4' in type_:
            time_hop = 0.005
            duration = 0.005
        else:
            time_hop = 0.02
            duration = 0.02
        layer = type_.split("_")[2]

        fname = os.path.join(
            stimuli_dir,
            layer,
            filename + '.npy',
        )

        spect = np.load(fname)
        spect = spect.swapaxes(
            0, -2
        )  # the time dimension is always not the last one but the one before
        spect = spect.reshape(
            spect.shape[0], -1
        )  # we put time in first dimension and concat the rest
        dimension = spect.shape[1]
    elif ("deepspeech" in type_
    ):

        time_hop = 0.02
        duration = 0.02
        layer = '_'.join(type_.split("_")[2:])

        fname = os.path.join(
            stimuli_dir,
            layer,
            filename + '.npy',
        )

        spect = np.load(fname) # already time x dime
        dimension = spect.shape[1]
    print('dimension data', dimension)

    feat_to_return = spect
    onsets = [i*time_hop for i in range(int(feat_to_return.shape[0]))]
    durations = duration * np.ones(len(onsets), dtype=float)

    return onsets, durations, dimension, feat_to_return





def get_features(
    stimuli_path,
    feature_set="word_embedding",
    shuffle_files=False,
    random_input=(False, 0),
    task_wanted = '',
    language_part = 'EN'
):
    """
    General function to get the features with onset and
    durations of the events we want to make the regression
    with to predict the fMRI
    """

    if ("wav2vec" in feature_set
        or "melfilterbanks" in feature_set
        or "mfccs" in feature_set
        or "rms" in feature_set
        or 'deepspeech' in feature_set
    ):
        #print("in", feature_set)
        onsets, durations, dim, features_ = add_event_sound(
            stimuli_dir=stimuli_path,
            type_=feature_set,
            shuffle_files=shuffle_files,
            task_wanted = task_wanted,
            language_part=language_part
        )

        #print('features shape', features_.shape)

    else:
        print("The type of features you want does not exist")
        raise ValueError

    # if you want random features
    #print("random wanted", random_input)
    if random_input[0]:
        print("random", random_input[1])
        if random_input[1] == 0:
            features_ = np.random.random(np.asarray(features_).shape)
        else:
            features_ = np.random.random(
                (np.asarray(features_).shape[0], random_input[1])
            )
    # features = time x dimension
    return onsets, np.asarray(features_), durations
