import torch
import torch.nn as nn
import os
import numpy as np

from utils import Generator
from reader import Dataset

#model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CGN(nn.Module):
    def __init__(self, config):
        super(CGN, self).__init__()
        self.dataset_dir = os.path.join(os.path.join(os.getcwd(), '../../dataset'), config["data_dir"])
        self.dataset = Dataset(self.dataset_dir)
        self.edim = config["edim"]
        self.num_items = config["num_items"]
        self.lr = config["lr"]
        self.lambda_cyc = config["lambda_cyc"]
        self.num_epoch = config["num_epoch"]
        self.batch_size = config["batch_size"]
        self.pretrained = config["pretrained"]

        self.init_generators()
        self.criterion = nn.MSELoss()
        self.init_embeddings()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def init_generators(self):
        self.g_network = Generator(self.num_items, self.edim).to(device)
        self.f_network = Generator(self.num_items, self.edim).to(device)

        #self.g_network_optimizer = torch.optim.Adam(self.g_network.parameters(), lr=self.lr)
        #self.f_network_optimizer = torch.optim.Adam(self.f_network.parameters(), lr=self.lr)
    
    def init_embeddings(self):
        if self.pretrained:
            U_d1 = torch.FloatTensor(np.load(os.path.join(self.dataset_dir, "clothing_user_emb.npy")))
            V_d1 = torch.FloatTensor(np.load(os.path.join(self.dataset_dir, "clothing_item_emb.npy")))
            U_d2 = torch.FloatTensor(np.load(os.path.join(self.dataset_dir, "homes_user_emb.npy")))
            V_d2 = torch.FloatTensor(np.load(os.path.join(self.dataset_dir, "homes_item_emb.npy")))
            self.U_d1 = nn.Embedding.from_pretrained(U_d1, freeze=True)
            self.U_d2 = nn.Embedding.from_pretrained(U_d2, freeze=True)
            self.V_d1 = nn.Embedding.from_pretrained(V_d1, freeze=True)
            self.V_d2 = nn.Embedding.from_pretrained(V_d2, freeze=True)
            print("Pretrained embeddings loaded and freezed!")
        else:
            self.U_d1 = nn.Embedding(self.dataset.num_user, self.edim, max_norm=1)
            self.U_d2 = nn.Embedding(self.dataset.num_user, self.edim, max_norm=1)
            self.V_d1 = nn.Embedding(self.dataset.num_item_d1, self.edim, max_norm=1)
            self.V_d2 = nn.Embedding(self.dataset.num_item_d2, self.edim, max_norm=1)
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def mmd_rbf(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        source = source.view(-1, self.num_items, self.edim)
        target = target.view(-1, self.num_items, self.edim)
        batch_size = int(source.size()[1])
        loss_all = []
        for i in range(int(source.size()[0])):
            kernels = self.guassian_kernel(source[i], target[i], kernel_mul=kernel_mul, kernel_num=kernel_num,
                                      fix_sigma=fix_sigma)
            xx = kernels[:batch_size, :batch_size]
            yy = kernels[batch_size:, batch_size:]
            xy = kernels[:batch_size, batch_size:]
            yx = kernels[batch_size:, :batch_size]
            loss = torch.mean(xx + yy - xy - yx)
            loss_all.append(loss)
        return sum(loss_all) / len(loss_all)

    def forward(self, user, item_d1, item_d2):
        user_d1 = self.U_d1(torch.LongTensor(user)).to(device)
        #print("user_emb:", user_d1.shape)
        user_d2 = self.U_d2(torch.LongTensor(user)).to(device)
        item_d1 = self.V_d1(torch.LongTensor(item_d1)).to(device)
        #print("item_emb:", item_d1.shape)
        item_d2 = self.V_d2(torch.LongTensor(item_d2)).to(device)

        pred_g = self.g_network.forward(item_d1, user_d2).view(-1, self.edim)
        #print("pred_g:", pred_g.shape)
        pred_f = self.f_network.forward(item_d2, user_d1).view(-1, self.edim)

        pred_fg = self.f_network.forward(self.g_network.forward(item_d1, user_d2).view(-1, self.edim), user_d1).view(-1, self.edim)
        pred_gf = self.g_network.forward(self.f_network.forward(item_d2, user_d1).view(-1, self.edim), user_d2).view(-1, self.edim)

        #print("pred_g", pred_g)
        #print("pred_f", pred_f)
        #print("pred_fg", pred_fg)
        #print("pred_gf", pred_gf)

        return pred_g, pred_f, pred_fg, pred_gf, item_d1, item_d2
    
    def evaluate_test(self):
        user, item_d1, item_d2 = self.dataset.create_test_iterable(self.num_items)
        user, item_d1, item_d2 = torch.tensor(user), torch.tensor(item_d1), torch.tensor(item_d2)

        pred_g, pred_f, pred_fg, pred_gf, actual_d1, actual_d2 = self.forward(user, torch.squeeze(item_d1.view(1,-1)), torch.squeeze(item_d2.view(1,-1)))
        pred_g, pred_f = pred_g.view(-1, self.num_items, self.edim).cpu().detach().numpy(), pred_f.view(-1, self.num_items, self.edim).cpu().detach().numpy()

        emb_d1 = self.V_d1(torch.LongTensor([i for i in range(self.dataset.num_item_d1)])).cpu().detach().numpy()
        emb_d2 = self.V_d2(torch.LongTensor([i for i in range(self.dataset.num_item_d2)])).cpu().detach().numpy()

        mapped_d1 = {}
        mapped_d2 = {}
        i=0
        for u in user:
            mapped_d2[u.item()] = self.dataset.item_mapping(pred_g[i], emb_d2)
            mapped_d1[u.item()] = self.dataset.item_mapping(pred_f[i], emb_d1)
            i+=1
        #print(mapped_d1)
        #print(mapped_d2)
        HR_d1, HR_d2, precision_d1, precision_d2, recall_d1, recall_d2 = self.dataset.evaluate(mapped_d1, mapped_d2)
        print("HR_d1", HR_d1)
        print("HR_d2", HR_d2)
        print("Precision_d1", precision_d1)
        print("Precision_d2", precision_d2)
        print("Recall_d1", recall_d1)
        print("Recall_d2", recall_d2)
        print()

    def train(self, num_epoch, Lambda):
        for epoch in range(num_epoch):
            print("Epoch {}".format(epoch+1))
            print("----------")
            user, item_d1, item_d2 = self.dataset.create_train_iterable(self.num_items)
            user, item_d1, item_d2 = torch.tensor(user), torch.tensor(item_d1), torch.tensor(item_d2)
            #print(item_d1.shape)
            
            permutation = torch.randperm(user.shape[0])
            l = (len(permutation)//self.batch_size - 1) * (self.batch_size)
            for batch in range(self.batch_size):
                self.optimizer.zero_grad()
                indices = permutation[batch : batch + self.batch_size]
                #print("user:", user[indices].shape)
                #print("item_d1:", item_d1[indices].shape)
                #print("item_d2:", item_d2[indices].shape)
                #print("item_d1:", torch.squeeze(item_d1[indices].view(1,-1)).shape)
                #print("item_d2:", torch.squeeze(item_d2[indices].view(1,-1)).shape)
                pred_g, pred_f, pred_fg, pred_gf, actual_d1, actual_d2 = self.forward(user[indices], torch.squeeze(item_d1[indices].view(1,-1)), torch.squeeze(item_d2[indices].view(1,-1)))
                
                loss_g = self.mmd_rbf(actual_d2, pred_g)
                loss_f = self.mmd_rbf(actual_d1, pred_f)
                loss_fg = self.criterion(actual_d1, pred_fg)
                loss_gf = self.criterion(actual_d2, pred_gf)
                #loss = loss_g + loss_f + self.lambda_cyc * (loss_fg + loss_gf)
                loss = loss_g + loss_f + Lambda * (loss_fg + loss_gf)
                loss.backward()
                self.optimizer.step()
            
            print("Loss_g: {:.4f}".format(loss_g))
            print("Loss_f: {:.4f}".format(loss_f))
            print("Loss_fg: {:.4f}".format(loss_fg))
            print("Loss_gf: {:.4f}".format(loss_gf))
            print("Loss_total: {:.4f}\n".format(loss))
            
            self.evaluate_test()
    
