from utils.preprocessing import DataSplitter

#script in cui si chiama il metodo di DataSplitter
# e si producono effettivamente i due csv splitted in training_raw
# e test_raw


DataSplitter.split("data/parkinsons_raw.data", 0.2)
