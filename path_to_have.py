#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" List of path we need"""

from pathlib import Path


# where the PetitPrince dataset is
DATA_PATH = Path('openneuro/LePetitPrince')
# where the mask will be saved
CACHE_PATH = Path('mask/')

# wav file path: useful only if you compute melfilterbanks (not in our case)
stimuli_path_wav = "stimuli_english_pp_16000/"


# Useful to get onset/offsets etc
path_to_english_data = "petit_prince/english/"
path_to_french_data = "petit_prince/french/"
path_to_mandarin_data = "petit_prince/mandarin/"

# self-supervised models
path_wav2vec_english = "wav2vec_english/"

