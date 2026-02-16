from utils.preprocessing import DataSplitter

#splitting the raw csv in non-test data and test data
DataSplitter.split("data/parkinsons_raw.data", 0.2)
