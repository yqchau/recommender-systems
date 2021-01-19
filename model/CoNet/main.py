from model import CoNet

config = {
    "data_dir": "books_and_movies",
    "domain_1": "movies",
    "domain_2": "books",
    "lr": 0.001,
    "edim": 32, 
    "cross_layer": 2,
    "reg": 0.0001,
    "batch_size": 32,
    "std": 0.01
}
model = CoNet(config, print_summary=True)
model.fit()
