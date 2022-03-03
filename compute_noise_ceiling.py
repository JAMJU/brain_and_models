#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Script to compute noise ceiling for each group of participant
"""

import numpy as np
from subject_data import Subjects
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from utils import correlate
from sklearn.decomposition import PCA
import os

def load_sub(sub, subjects):
    data = subjects(subject_id=sub, run_nb=1)
    for run in range(2, 10):
        data = np.concatenate((data, subjects(subject_id=sub, run_nb=run)), axis=0)
    return data


def load_all_fmri_(list_subjects, subjects):

    fmri_all_subjects = load_sub(sub=list_subjects[0], subjects=subjects)

    for sub in list_subjects[1:]:
        print(sub)
        fmri_all_subjects += load_sub(sub=sub, subjects=subjects)
    fmri_all_subjects = fmri_all_subjects/float(len(list_subjects))
    return fmri_all_subjects


def compute_noise_ceiling(folder, language):
    print('Entered')
    subjects = Subjects(lang=language)
    list_subjects = subjects.files.subject.unique()
    print(list_subjects)

    #subjects = load_all_fmri_data(language = language)
    #print('loaded all subjects')
    R = []
    for subject_id, subject_name in enumerate(list_subjects):
        file_subject = folder + language + '_' + str(subject_id) + '.npy'
        if subject_name in ['sub-CN018', 'sub-CN019']:
            continue
        if os.path.isfile(file_subject):
            R.append(np.load(file_subject))
        else:
            print('for', subject_name)
            #subject = subjects[subject_id]
            # we apply a pca
            pca_fmri = PCA(n_components=100)
            Y = pca_fmri.fit_transform(load_sub(sub = subject_name, subjects = subjects))
            print("pca subject", subject_id, "computed")
            #subjects_but_one =
            #avg = np.mean(subjects_but_one, 0)

            pca_av = PCA(n_components=100)
            X = pca_av.fit_transform(load_all_fmri_([sub for sub in list_subjects if (sub != subject_name and sub not in  ['sub-CN018', 'sub-CN019'])], subjects=subjects))
            print("pca other subjects than", subject_id, "computed")

            print(X.shape, Y.shape)

            assert Y.shape == X.shape

            # predict PCA(subject) from average across subjects (no time lag)
            cv = KFold(10, shuffle=False)
            y_pred = cross_val_predict(RidgeCV(), X=X, y=Y, cv=cv)
            y_pred = pca_fmri.inverse_transform(y_pred)
            r = correlate(load_sub(sub = subject_name, subjects = subjects), y_pred)
            np.save(file_subject, r)
            R.append(r)
    return np.asarray(R)


def compute_and_save(language, fileout, folder):
    print('in it')
    np.save(fileout, compute_noise_ceiling(language=language, folder = folder))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='script to launch analysis subject per subject')
    parser.add_argument('language', metavar='subject', type=str,
                        help='subject to analyze')

    args = parser.parse_args()
    if args.language == 'EN':
        compute_and_save(language='EN', fileout='english.npy', folder = 'petit_prince/noise_ceiling/') # you need to change the path
        print('English done')
    elif args.language == 'FR':
        compute_and_save(language='FR', fileout='french.npy', folder = 'petit_prince/noise_ceiling/')
        print('French done')
    elif args.language == 'CN':
        compute_and_save(language='CN', fileout='chinese.npy', folder = 'petit_prince/noise_ceiling/')
        print('Mandarin done')