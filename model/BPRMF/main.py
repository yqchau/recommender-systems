from model import BPRMF

config = {
    "data_dir": "books_and_movies",
    "domain": "books",
    "lr": 0.001,
    "edim": 8, 
    "reg": 0.0001,
    "batch_size": 32,
}
model = BPRMF(config, print_summary=True)
print()
model.fit()
