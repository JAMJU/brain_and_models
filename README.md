# brain_and_models
This repo is still under construction, please contact us if you have any question (juliette.millet@cri-paris.org)


To use this repo, you need to :
- change the paths in path_to_have.py (in particular the path to the dataset )
- transform the wavfile you want to use: first convert them to 16000hz and then use models to extract their representation
- If you are using a wav2vec model, copy what is done for wav2vec_english. The representation of wav2vec needs to be in one folder, with one subfolder per layer with the following names: conv4, conv5, conv6, transf0, transf1 etc... Follow the instructions fowav2vec extraction in : https://github.com/JAMJU/Sel_supervised_models_perception_biases to have the right format f representation.

Contact us if you want to have access to the models we trained (juliette.millet@cri-paris.org)


## Launching experiments

### On one participant
to launch the regression process on the English participant sub-EN073, do:


`python main.py sub-EN073 folder_results wav2vec_english EN`

### On multiple participants
create_sbatch.py enables you to create sbatch file for all participants in a language group

### Noise ceiling
The code to compute the noise ceiling is available in noise_ceiling.py

## Organization

main.py: call of the main function # no need to change this (except with the language option)

parallel_regression.py: regression functions # you will probably have to change the get_feat_fmri function (for the fmri part)

features.py: get features # you will have to change some details (line 19 to 24) to adapt to a new dataset

subject_data.py: get fmri data # It is mainly this file that you need to change to adapt this code to a new dataset

regressor_hier.py: regressor definition # no need to change this 


