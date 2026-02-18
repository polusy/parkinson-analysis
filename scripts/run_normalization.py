from utils.preprocessing import DataNormalizer


#normalizing training and test (unbalanced) parkinsons data
DataNormalizer.normalize_data("data/unbalanced/raw/raw_parkinsons_training.data", "data/unbalanced/raw/raw_parkinsons_test.data", "data/unbalanced/normalized/normalized_parkinsons_training.data", "data/unbalanced/normalized/normalized_parkinsons_test.data")


#normalizing training and test (balanced) parkinsons data
DataNormalizer.normalize_data("data/balanced/raw/balanced_parkinsons_training.data", "data/balanced/raw/balanced_parkinsons_test.data", "data/balanced/normalized/normalized_balanced_parkinsons_training.data", "data/balanced/normalized/normalized_balanced_parkinsons_test.data")
