import torch.nn as nn
import torch.nn.functional as F
import torch


class DropNet(nn.Module):
    def __init__(self, input_dim):
        super(DropNet, self).__init__()

        self.l1 = nn.Linear(input_dim, 4096)

        self.l2 = nn.Linear(4096, 2048)

        self.u1 = nn.Linear(2048, 4096)

        self.u2 = nn.Linear(4096, input_dim)

        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(2048)


        self.dn1 = nn.BatchNorm1d(4096)
        self.dn2 = nn.BatchNorm1d(input_dim)

        self.sig = nn.Sigmoid()

    def initializd(self):
        for m in self.modules():
            m.weight.data = nn.init.xavier_uniform_(m.weight)

    def forward(self, x, mask_ratio):
        x, mask = self.random_masking(x, mask_ratio)

        # with BN
        x = F.relu(self.bn1(self.l1(x)))

        x = F.relu(self.bn2(self.l2(x)))


        f = x

        x = F.relu(self.dn1(self.u1(x)))

        x = F.relu(self.dn2(self.u2(x)))

        return x, f

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.ones(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = x
        x_masked[:,~ids_keep] = noise[:,~ids_keep]

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        return x_masked, mask
    
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(input_dim, 4096)
        self.l1_ = nn.Linear(4096, 4096)

        self.l2 = nn.Linear(4096, 2048)
        self.l2_ = nn.Linear(2048, 2048)

        self.l3 = nn.Linear(2048, 1024)
        self.l3_ = nn.Linear(1024, 1024)

        self.l4 = nn.Linear(1024, 512)
        self.l4_ = nn.Linear(512, 512)
        
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):


        x = F.relu(self.bn1(self.l1(x)))
        x = x + F.relu(self.bn1(self.l1_(x)))

        x = F.relu(self.bn2(self.l2(x)))
        x = x + F.relu(self.bn2(self.l2_(x)))

        x = F.relu(self.bn3(self.l3(x)))
        x = x + F.relu(self.bn3(self.l3_(x)))

        x = F.relu(self.bn4(self.l4(x)))
        x = x + F.relu(self.bn4(self.l4_(x)))
        
        x = self.sig(x)
        return x



class CLNet(nn.Module):
    def __init__(self, DropNet, Drop_input, classes):
        super(CLNet, self).__init__()

        self.pretrain = DropNet(Drop_input, requires_grad = False)
        self.f1 = nn.Linear(128, 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, 64)
        self.f4 = nn.Linear(64, classes)
    def forward(self, x):
        _, f = self.pretrain(x)
        f = F.relu(self.f1(f))
        f = F.relu(self.f2(f))
        f = F.relu(self.f3(f))
        f = F.log_softmax(self.f4(f), dim = 1)
        return f

