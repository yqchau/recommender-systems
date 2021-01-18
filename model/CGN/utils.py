#generator
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, num_items, item_dim):
        super(Generator, self).__init__()
        self.num_items = num_items
        self.item_dim = item_dim

        self.fc1 = nn.Linear((self.num_items + 1) * self.item_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, self.num_items * self.item_dim)

    def forward(self, other_domain_items, user_emb):
        other_domain_items = other_domain_items.view(-1, self.num_items * self.item_dim)
        #print("other_domain_items", other_domain_items)
        user_emb = user_emb.view(-1, self.item_dim)
        #print("user_emb", user_emb)

        out = torch.cat([other_domain_items, user_emb], 1)
        #print("out", out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        out = nn.Tanh()(out)
        return out
