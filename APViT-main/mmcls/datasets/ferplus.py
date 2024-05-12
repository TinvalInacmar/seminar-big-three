from .builder import DATASETS

from .base_fer_dataset import BaseFerDataset

@DATASETS.register_module()
class FERPlus(BaseFerDataset):
    DATASET_CLASSES = ["Neutral", "Happiness", "Other"]
    CONVERT_TABLE = (0, 1, 2, 2, 1, 2, 0, 2) 
#FER_BASE_CLASSES = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Happiness', 'Surprise', 'Neutral', 'Contempt']

