from model import CGN

config = {
    "data_dir": "clothing_and_homes",
    "edim": 10,
    "num_items": 5,
    "lr": 0.00005,
    "lambda_cyc": 0.5,
    "num_epoch": 100,
    "batch_size": 64,
    "pretrained": True
}

model = CGN(config)
print()
model.train()
