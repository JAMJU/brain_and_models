#!/usr/bin/env python
# -*- coding: utf-8 -*-

from parallel_regression import main
import numpy as np
import os

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to launch analysis subject per subject')
    parser.add_argument('subject', metavar='subject', type=str,
                        help='subject to analyze')
    parser.add_argument('folder_save', metavar='fold', type=str,
                        help='folder where to save the results')
    parser.add_argument('model', metavar='fold', type=str,
                        help='model to use')
    parser.add_argument('language_part', metavar='fold', type=str,
                        help='language of the participants to use')



    args = parser.parse_args()

    model = "glover"
    estimator_ = "ridge"

    # the layers we are using for wav2vec
    sel_features = ['wav2vec_english_conv4', 'wav2vec_english_conv5', 'wav2vec_english_conv6','wav2vec_english_transf0',
                    'wav2vec_english_transf1', 'wav2vec_english_transf2',
                    'wav2vec_english_transf3', 'wav2vec_english_transf4',
                    'wav2vec_english_transf5', 'wav2vec_english_transf6',
                    'wav2vec_english_transf7', 'wav2vec_english_transf8',
                    'wav2vec_english_transf9', 'wav2vec_english_transf10',
                    'wav2vec_english_transf11']



    # depending on the model used, we change the name of teh layers used
    if args.model == 'wav2vec_english':
        sel_features = sel_features
    else:
        print('model unavailable')
    hierarch = dict(subtract=False, concat=True)
    scale_features = True  # keep that
    scale_before = ['minmax' for i in range(25)]  # keep that
    random_input = (False, 0)
    tempname = args.subject + '_' + args.model
    using_pca_fmri = True



    result, coeff_values = main(subject=args.subject,
        model=model,
        estimator_=estimator_,
        sel_features=sel_features,
        hierarch=hierarch,
        scale_features=scale_features,
        scale_before=scale_before,
        random_input=random_input,
        tempname=os.path.join(args.folder_save,tempname),
        language_part=args.language_part,
        using_pca_fmri=using_pca_fmri
         )

    np.save(os.path.join(args.folder_save, args.subject + '.npy'), result)
    np.save(os.path.join(args.folder_save, args.subject + '_coeffs.npy'), coeff_values )

