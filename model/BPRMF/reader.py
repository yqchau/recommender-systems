import os
import pandas as pd
import numpy as np
import random

def load_file(file_name):
    df = pd.read_csv(file_name)
    user = df["users"]
    item = df["items"]

    actual_dict = {}
    for user, sf in df.groupby("users"):
        actual_dict[user] = list(sf["items"])

    data = df[["users", "items"]].to_numpy(dtype=int).tolist()
    return data, actual_dict, set(df["users"]), set(df["items"])

class Dataset:

    def __init__(self, path, name, print_summary=False):
        if name == "books":
            self.train_path = os.path.join(path, "books_train.csv")
            self.test_path = os.path.join(path, "books_test.csv")
        
        elif name == "movies":
            self.train_path = os.path.join(path, "movies_train.csv")
            self.test_path = os.path.join(path, "movies_test.csv")

        self.print_summary = print_summary
        self.initialize()
    
    def initialize(self):
        self.train_data, self.train_dict, train_user_set, train_item_set = load_file(self.train_path)
        self.test_data, self.test_dict, test_user_set, test_item_set = load_file(self.test_path)

        assert (test_user_set.issubset(train_user_set))
        assert (test_item_set.issubset(train_item_set))
        self.user_set = train_user_set
        self.item_set = train_user_set
        self.num_user = len(train_user_set)
        self.num_item = len(train_item_set)
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)

        self.test_input_dict, self.train_neg_dict = self.get_dicts()
        #self.check_dict_error(test_user_set)
        
        if self.print_summary:
            print("Train size:", self.train_size)
            print("Test size:", self.test_size)
            print("Number of user:", self.num_user)
            print("Number of item:", self.num_item)
            print("Data Sparsity: {:.1f}%".format(100 * (self.num_user * self.num_item - self.train_size)/ (self.num_user * self.num_item)))
    
    def get_dicts(self):
        train_actual_dict, test_actual_dict = self.train_dict, self.test_dict
        train_neg_dict = {}
        test_input_dict = {}
        for user in list(self.user_set):
            train_neg_dict[user] = list(self.item_set - set(train_actual_dict[user]))

        for user in test_actual_dict.keys():
            test_input_dict[user] = train_neg_dict[user]
            train_neg_dict[user] = list(set(train_neg_dict[user]) - set(test_actual_dict[user]))
    
        return test_input_dict, train_neg_dict

    def neg_sampling(self, num):
        item_dict = self.train_neg_dict
        user_list = []
        item_list = []
        #print(item_dict)
        for user in list(self.user_set):
            items = random.sample(item_dict[user], 20)
            item_list += items
            user_list += [user] * len(items)
        #print(user_list)
        #print(item_list)
        result = np.transpose(np.array([user_list, item_list]))
        return random.sample(result.tolist(), num)

    def get_train(self):
        neg = self.neg_sampling(num=self.train_size)
        pos = self.train_data
        df_pos = pd.DataFrame(pos,columns=['users', 'items'])
        df_neg = pd.DataFrame(neg,columns=['users', 'items'])
        out = []
        for user, sf_pos in df_pos.groupby("users"):
            sf_neg = df_neg.loc[df_neg["users"] == user]
            if sf_neg.empty:
                continue
            for pos_item in sf_pos["items"].tolist():
                for neg_item in sf_neg["items"].tolist():
                    out.append([user, pos_item, neg_item])

        return out

    def get_data(self):
        return self.test_dict, self.test_input_dict, self.test_data, self.num_user, self.num_item
    
