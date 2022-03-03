from pathlib import Path
import pandas as pd
from nilearn.masking import compute_epi_mask
from nilearn.image import load_img, mean_img
from nilearn.masking import apply_mask
import nibabel as nib
from path_to_have import DATA_PATH, CACHE_PATH

class Subjects():
    """
    Class to get information from subjects
    """
    def __init__(self, lang = 'EN', path=DATA_PATH, cache=CACHE_PATH):
        """

        :param lang: EN (english) FR (french) or CN (Chinese)
        :param path: path to dataset
        :param cache: where the mask will be saved
        """
        path = Path(path)
        self.path = Path(path)
        self.cache = Path(cache)
        self.language = lang
        subjects = (path / 'derivatives').iterdir()
        files = list()
        for subject in subjects:
            if 'sub-' not in subject.name:
                continue

            runs = (subject / 'func').iterdir()
            language = subject.name.split('-')[1][:2]
            if language != self.language:
                continue
            for run_file in runs:
                #print(run_file.name)
                if not run_file.name.endswith('.nii.gz'):
                    continue
                run_id = int(run_file.name.split('-')[3][:2])
                file = dict(subject=subject.name,
                            run=int(run_id),
                            language=language,
                            file=run_file)
                files.append(file)
        self.files = pd.DataFrame(files)
        #print(self.files)

    def __call__(self, subject_id, run_nb):
        """ Gets the fmri masked for one specific run """
        files = self.files[self.files['subject'] == subject_id]
        files = files.sort_values(by = 'run')
        #print(files)
        files = files.iloc[int(run_nb - 1)]
        #for file in files.file:
        return self._read_fmri(str(files.file))

    @property
    def mask(self):
        """ Computes mask for language group """
        if not (self.cache / 'mask.nii').exists():
            img = mean_img([str(f) for f in self.files.file])
            mask = compute_epi_mask(img)
            nib.save(mask, self.cache / 'mask.nii')
        return nib.load(self.cache / 'mask.nii')

    def _read_fmri(self, file):
        """ Read fmri from file """
        imgs = load_img(str(file))
        masked_imgs = apply_mask(imgs, self.mask)
        return masked_imgs