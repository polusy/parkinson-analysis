from utils.preprocessing import DataNormalizer


#normalizing training and test parkinsons data
DataNormalizer.normalize_data("data/raw/raw_parkinsons_training.data", "data/raw/raw_parkinsons_test.data")


