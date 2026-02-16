from utils.preprocessing import DataNormalizer


#normalizing training and test parkinsons data
DataNormalizer.normalize_data("data/raw_parkinsons_training.data", "data/raw_parkinsons_test.data")


