import torch
import torch.nn.functional as F
from Model_CLNet import *
import torch.optim as optim
import numpy as np
import os
import logging
import torch.nn as nn
from preprocess import *
from Gene_dataset import *
import time
dir_checkpoint = './CLNet_checkpoint/'

def load_checkpoint(model, checkpoint_PATH):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT)
    return model

GeneHeadDataset = GeneHeadDataset()
GeneMediumDataset = GeneMediumDataset()
GeneTailDataset = GeneTailDataset()

def train_net(
        pretrain_net,
        net_head,
        net_medium,
        net_tail,
        device,
        epochs=5,
        batch_size=3,
        lr_head = 0.0001,
        lr_medium = 0.001,
        lr_tail = 0.01,
        save_cp=True
        ):


    optimizer_head = optim.Adam(net_head.parameters(), lr=lr_head)
    optimizer_medium = optim.Adam(net_medium.parameters(), lr=lr_medium)
    optimizer_tail = optim.Adam(net_tail.parameters(), lr=lr_tail)
    optimizer_pretrain = optim.Adam(pretrain_net.parameters(), lr=lr_head)

    #np.random.seed(201)

    for epoch in range(epochs):
        net_tail.train()
        net_medium.train()
        net_head.train()
        pretrain_net.train()

        # Head Classes Training
        dataloader_Head = DataLoader(GeneHeadDataset, batch_size=128, shuffle=True, num_workers=8)
        dataloader_Medium = DataLoader(GeneMediumDataset, batch_size=64, shuffle=True, num_workers=8)
        dataloader_Tail = DataLoader(GeneTailDataset, batch_size=32, shuffle=True, num_workers=8)
        Loss_Head = 0

        for index, (data1, data2, data3) in enumerate(
                zip(dataloader_Head, dataloader_Medium, dataloader_Tail)):
            batch_data = torch.from_numpy(np.concatenate((data1[0], data2[0], data3[0]), axis=0)).to(device=device,dtype=torch.float32)
            batch_label = torch.from_numpy(np.concatenate((data1[1], data2[1], data3[1]), axis=0)).to(device=device,dtype=torch.long)
            batch_rec, f = pretrain_net(batch_data)
            output = net_head(f)

            _, f_rec = pretrain_net(batch_rec)
            output_rec = net_head(f_rec)

            loss_head = F.nll_loss(output, batch_label)

            Loss_Head += loss_head.detach()

            optimizer_head.zero_grad()
            optimizer_pretrain.zero_grad()

            loss_head.backward()
            optimizer_head.step()
            optimizer_pretrain.step()



        # Medium Classes Training
        dataloader_Head = DataLoader(GeneHeadDataset, batch_size=64, shuffle=True, num_workers=8)
        dataloader_Medium = DataLoader(GeneMediumDataset, batch_size=64, shuffle=True, num_workers=8)
        dataloader_Tail = DataLoader(GeneTailDataset, batch_size=64, shuffle=True, num_workers=8)
        Loss_Medium = 0
        for index, (data1, data2, data3) in enumerate(
                zip(dataloader_Head, cycle(dataloader_Medium), cycle(dataloader_Tail))):
            batch_data = torch.from_numpy(np.concatenate((data1[0], data2[0], data3[0]), axis=0)).to(device=device,dtype=torch.float32)
            batch_label = torch.from_numpy(np.concatenate((data1[1], data2[1], data3[1]), axis=0)).to(device=device,dtype=torch.long)
            batch_rec, f = pretrain_net(batch_data)
            output = net_medium(f)

            _, f_rec = pretrain_net(batch_rec)
            output_rec = net_head(f_rec)

            loss_medium = F.nll_loss(output, batch_label)

            Loss_Medium += loss_medium.detach()

            optimizer_pretrain.zero_grad()
            optimizer_medium.zero_grad()

            loss_medium.backward()

            optimizer_pretrain.step()
            optimizer_medium.step()

        # Tail Classes Training
        dataloader_Head = DataLoader(GeneHeadDataset, batch_size=32, shuffle=True, num_workers=8)
        dataloader_Medium = DataLoader(GeneMediumDataset, batch_size=64, shuffle=True, num_workers=8)
        dataloader_Tail = DataLoader(GeneTailDataset, batch_size=128, shuffle=True, num_workers=8)
        Loss_Tail = 0
        for index, (data1, data2, data3) in enumerate(
                zip(dataloader_Head, cycle(dataloader_Medium), cycle(dataloader_Tail))):
            batch_data = torch.from_numpy(np.concatenate((data1[0], data2[0], data3[0]), axis=0)).to(device=device,dtype=torch.float32)
            batch_label = torch.from_numpy(np.concatenate((data1[1], data2[1], data3[1]), axis=0)).to(device=device,dtype=torch.long)
            batch_rec, f = pretrain_net(batch_data)
            output = net_tail(f)

            _, f_rec = pretrain_net(batch_rec)
            output_rec = net_head(f_rec)

            loss_tail = F.nll_loss(output, batch_label)
            Loss_Tail += loss_tail.detach()

            optimizer_tail.zero_grad()
            optimizer_pretrain.zero_grad()

            loss_tail.backward()
            optimizer_pretrain.zero_grad()
            optimizer_tail.step()

        print("Epoch [",epoch,"] Loss Head: ", Loss_Head.item(), " Loss Medium: ", Loss_Medium.item(), " Loss Tail: ", Loss_Tail.item())

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if epoch % 50 == 0:
                torch.save(pretrain_net.state_dict(), f'./checkpoint/CP_epoch{epoch+1}.path')
                torch.save(net_head.state_dict(), dir_checkpoint + f'Head_epoch{epoch + 1}.path')
                torch.save(net_medium.state_dict(), dir_checkpoint + f'Medium_epoch{epoch + 1}.path')
                torch.save(net_tail.state_dict(), dir_checkpoint + f'Tail_epoch{epoch + 1}.path')
                logging.info(f'Checkpoint {epoch + 1} saved')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = 15254
    gene_class = 10
    checkpoint = './checkpoint/CP_epoch2001.path'

    pretrain_net = DropNet_(input_dim)
    pretrain_net.cuda()
    #pretrain_net = nn.DataParallel(pretrain_net, device_ids=[1])
    #pretrain_net = load_checkpoint(pretrain_net, checkpoint)

    net_head = CLNet(gene_class)
    net_head.to(device=device)
    net_head.cuda()
    #net_head = nn.DataParallel(net_head, device_ids=[1])

    net_medium = CLNet(gene_class)
    net_medium.to(device=device)
    net_medium.cuda()
    #net_medium = nn.DataParallel(net_medium, device_ids=[1])

    net_tail = CLNet(gene_class)
    net_tail.to(device=device)
    net_tail.cuda()
    #net_tail = nn.DataParallel(net_tail, device_ids=[1])


    train_net(
        pretrain_net,
        net_head,
        net_medium,
        net_tail,
        epochs=401,
        batch_size=400,
        device=device,
        save_cp=True,
    )
