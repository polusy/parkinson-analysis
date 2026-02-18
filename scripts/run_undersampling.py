from utils.preprocessing import DataUndersampler

#undersampling positives data in the dataset
DataUndersampler.undersample_positives("data/unbalanced/raw/parkinsons_raw.data", "data/balanced/raw/balanced_parkinsons_raw.data")

