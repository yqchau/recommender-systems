import pandas as pd

def train_iterable(train_d1, train_d2, label_d1, label_d2):
    df_d1 = pd.DataFrame(train_d1, columns=["users_d1", "items_d1"])
    df_d2 = pd.DataFrame(train_d2, columns=["users_d2", "items_d2"])
    df_d1["label_d1"] = label_d1
    df_d2["label_d2"] = label_d2
    train_frames = []
    for user, sf_d1 in df_d1.groupby("users_d1"):
        df = pd.DataFrame()
        sf_d2 = df_d2.loc[df_d2["users_d2"] == user]
        l = min(len(sf_d1), len(sf_d2))
        #print(l)
        df["users"] = l*[user]
        #print(sf_d2)
        #print(sf_d2["items"].sample(n=1))
        sample_d1 = sf_d1.sample(n=l)
        #print(sample_d1)
        sample_d2 = sf_d2.sample(n=l)
        #print(sample_d2)
        df["items_d1"] = sample_d1["items_d1"].values
        df["items_d2"] = sample_d2["items_d2"].values
        df["labels_d1"] = sample_d1["label_d1"].values
        df["labels_d2"] = sample_d2["label_d2"].values
        #print(df)
        train_frames.append(df)
    frame = pd.concat(train_frames)
    data = frame[["users", "items_d1", "items_d2"]].values.tolist()
    labels = frame[["labels_d1", "labels_d2"]].values.tolist()
    return data, labels
