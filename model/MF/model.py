from reader import Dataset
import os
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MF(nn.Module):

    def __init__(self, config, print_summary=False):
        super(MF, self).__init__()
        self.print_summary = print_summary
        new_path = os.path.join(os.path.join(os.getcwd(), '../../dataset'), config["data_dir"])
        self.dataset = Dataset(new_path, config["domain"], self.print_summary)
        self.test_dict, self.test_input_dict, self.test_data, self.num_user, self.num_item = self.dataset.get_data()

        self.edim = config["edim"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.batch_size = config["batch_size"]

        self.U = nn.Embedding(self.num_user, self.edim)
        self.V = nn.Embedding(self.num_item, self.edim)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user, item):
        user_emb = self.U(torch.LongTensor(user)).to(device=device)
        item_emb = self.V(torch.LongTensor(item)).to(device=device)
        return self.sigmoid(torch.sum(torch.mul(user_emb, item_emb), 1))
    
    def fit(self, num_epoch=101):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        criterion = nn.MSELoss()
        data, labels = self.dataset.get_train()
        train_data = torch.tensor(data)
        labels = torch.tensor(labels, device=device)
        user, item = train_data[:,0], train_data[:,1]
        for epoch in range(num_epoch):
            permutation = torch.randperm(user.shape[0])
            max_idx = int((len(permutation) // (self.batch_size/2) -1) * (self.batch_size/2))
            for batch in range(0, max_idx, self.batch_size):
                optimizer.zero_grad()
                idx = permutation[batch : batch + self.batch_size]  
                output = self.forward(user[idx], item[idx])
                #print(labels[idx])
                #print(output)
                loss = criterion(labels[idx].float(), output)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:    
                print("epoch {} loss: {:.4f}".format(epoch, loss))
                self.evaluate_test_at_k(10)

    def evaluate_test_at_k(self, k=10):
        with torch.no_grad():
            relevant = 0
            selected = 0
            hit = 0
            n = 0
            for user in self.test_dict.keys():
                items = torch.tensor(self.test_input_dict[user])
                users = torch.tensor([user] * len(items))
                output = self.forward(users, items)
                #print(output)
                indices = torch.argsort(output, dim=0, descending=True)[0:k].tolist()
                #print(indices)
                pred = []
                for idx in indices:
                    pred.append(items[idx])
                actual = self.test_dict[user]
                #print(pred)
                #print(actual)
                reward = 0
                for item in pred:
                    if item in actual:
                        reward += 1
                
                n+=reward
                relevant += len(actual)
                selected += len(pred)
                if reward > 0:
                    hit += 1
            
            #print("HIT:", hit)
            #print("n:", n)
            print("HIT RATIO@{}: {:.4f}".format(k,hit/len(self.test_dict.keys())))
            print("PRECISION@{}: {:.4f}".format(k, n/selected))
            print("RECALL@{}: {:.4f}".format(k, n/relevant))
            print()


