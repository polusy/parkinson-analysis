from utils.preprocessing import DataSplitter

#splitting the raw csv in non-test data and test data
DataSplitter.split("data/unbalanced/raw/parkinsons_raw.data", "data/unbalanced/raw/raw_parkinsons_training.data", "data/unbalanced/raw/raw_parkinsons_test.data",  0.2)


#splitting the raw csv in non-test data and test data
DataSplitter.split("data/balanced/raw/balanced_parkinsons_raw.data", "data/balanced/raw/balanced_parkinsons_training.data", "data/balanced/raw/balanced_parkinsons_test.data",  0.2)