#Dataset
import os
import pandas as pd
import numpy as np

def load_train_file(file_name, sep=" "):
    df = pd.read_csv(file_name, sep=sep, names=["users", "items", "ratings", "partition"])
    print("Train set size = ", len(df))
    user_dict = {}

    for name, g in df.groupby("users"):
        temp_user_dict = {}
        for part, p in g.groupby("partition"):
            movies = p["items"].values
            temp_user_dict[part] = movies.tolist()
        user_dict[name] = temp_user_dict

    return user_dict, set(df["users"]), set(df["items"])

def load_test_file(file_name, sep=" "):
    df = pd.read_csv(file_name, sep=sep, names=["users", "items", "ratings", "partition"])
    print('Test set size = ', len(df))
    user_dict = {}

    for name, g in df.groupby("users"):
        movies = g["items"].values
        user_dict[name] = movies.tolist()
    return user_dict

class Dataset:

    def __init__(self, path):
        #domain 1
        self.train_path_d1 = os.path.join(path, "clothing_train.txt")
        self.test_path_d1 = os.path.join(path, "clothing_test.txt")

        #domain 2
        self.train_path_d2 = os.path.join(path, "homes_train.txt")
        self.test_path_d2 = os.path.join(path, "homes_test.txt")

        self.initialize()
    
    def initialize(self):
        #load train data
        self.train_dict_d1, user_set_d1, item_set_d1 = load_train_file(self.train_path_d1)
        self.train_dict_d2, user_set_d2, item_set_d2 = load_train_file(self.train_path_d2)
        assert (len(user_set_d1) == len(user_set_d2))
        user_set = user_set_d1

        self.user_list = list(user_set)
        self.item_list_d1 = list(item_set_d1)
        self.item_list_d2 = list(item_set_d2)

        #load test data
        self.test_dict_d1 = load_test_file(self.test_path_d1)
        self.test_dict_d2 = load_test_file(self.test_path_d2)

        self.num_user = len(self.user_list)
        self.num_item_d1 = len(self.item_list_d1)
        self.num_item_d2 = len(self.item_list_d2)

        print("Number of user:", self.num_user)
        print("Number of item in domain1:", self.num_item_d1)
        print("Number of item in domain2:", self.num_item_d2)
    
    def create_train_iterable(self, num_samples):
        users = []
        items_d1 = []
        items_d2 = []

        for user in range(self.num_user):
            train_d1 = self.train_dict_d1[user]
            train_d2 = self.train_dict_d2[user]
            num_partition = min(len(train_d1), len(train_d1))
            for partition in range(num_partition):
                users.append(user)
                items_d1.append(np.random.choice(train_d1[partition], num_samples).tolist())
                items_d2.append(np.random.choice(train_d2[partition], num_samples).tolist())
        
        #items_d1 = np.squeeze(np.array(items_d1).reshape(1,-1)).tolist()
        #items_d2 = np.squeeze(np.array(items_d2).reshape(1,-1)).tolist()

        return users, items_d1, items_d2
    
    def create_test_iterable(self, num_samples):
        users = list(set(self.test_dict_d1.keys()).intersection(set(self.test_dict_d2.keys())))
        #print(users)
        items_d1 = []
        items_d2 = []

        for user in users:
            items_d1.append(np.random.choice(self.test_dict_d1[user], num_samples).tolist())
            items_d2.append(np.random.choice(self.test_dict_d2[user], num_samples).tolist())
        
        return users, items_d1, items_d2

    def item_mapping(self, pred_emb, all_emb):
        index = []
        for emb in pred_emb:
            distance = np.sum((emb-all_emb)**2, axis=1)
            index.append(np.squeeze(np.where(distance == np.min(distance))).tolist())
        return index
    
    def evaluate(self, mapped_d1, mapped_d2):
        hit_d1, hit_d2= 0, 0
        assert (mapped_d1.keys() == mapped_d2.keys())

        len_actual_d1 = 0
        len_actual_d2 = 0
        
        reward_d1 = 0
        reward_d2 = 0

        for user in mapped_d1.keys():
            actual_d1 = self.test_dict_d1[user]
            actual_d2 = self.test_dict_d2[user]
            pred_d1  = mapped_d1[user]
            pred_d2 = mapped_d2[user]

            len_actual_d1 += len(actual_d1)
            len_actual_d2 += len(actual_d2)
            
            n_d1 = 0
            n_d2 = 0
            assert (len(pred_d1) == len(pred_d2))
            for i in range(len(pred_d1)):
                if pred_d1[i] in actual_d1:
                    n_d1+=1
                if pred_d2[i] in actual_d2:
                    n_d2+=1
            
            hit_d1 += min(n_d1, 1)
            hit_d2 += min(n_d2, 1)

            reward_d1 += n_d1
            reward_d2 += n_d2
        n_user = len(mapped_d1.keys())
        total_pred = n_user * len(pred_d1)

        #print("n_user", n_user)
        #print("total_pred", total_pred)
        #print("len_actual_d1", len_actual_d1)
        #print("len_actual_d2", len_actual_d2)
        #print("hit_d1", hit_d1)
        #print("hit_d2", hit_d2)
        #print("reward_d1", reward_d1)
        #print("reward_d2", reward_d2)
        HR_d1, HR_d2 = hit_d1/n_user, hit_d2/n_user
        precision_d1, precision_d2 = reward_d1/total_pred, reward_d2/total_pred
        recall_d1, recall_d2 = reward_d1/len_actual_d1, reward_d2/len_actual_d2
        return HR_d1, HR_d2, precision_d1, precision_d2, recall_d1, recall_d2




