import torch
import torch.nn.functional as F
from Model import *
import torch.optim as optim
import numpy as np
import os
import logging
from torch.autograd import Variable
from preprocess import *
dir_checkpoint = './checkpoint/'

def load_checkpoint(model, checkpoint_PATH):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT)
    return model


def train_net(
        net,
        dnet,
        device,
        epochs=5,
        batch_size=3,
        lr=0.001,
        save_cp=True
        ):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer_d = optim.Adam(dnet.parameters(), lr=lr)
    criterion_MSE = torch.nn.MSELoss()
    criterion_BCE = torch.nn.BCELoss()
    criterion_KL = torch.nn.KLDivLoss(reduction='batchmean')
    data = np.load('./Data/Train/x.npy')
    data = preprocess(data)

    np.random.seed(201)
    np.random.shuffle(data)

    for epoch in range(epochs):
        net.train()
        MSE_LOSS = 0
        KL_LOSS = 0
        Total_loss = 0
        D_loss = 0
        ind = 0
        lambda_KL = 1.0
        lambda_MSE = 1.0
        for i in range(0, data.shape[0], batch_size):
            batch_data = torch.from_numpy(data[i:i+batch_size]).to(device=device,dtype=torch.float32)
            batch_label = batch_data.detach().clone()

            output, _ = net(batch_data, 0.25) # 1-value = masked ratio


            MSE = criterion_MSE(output, batch_label)
            KL = criterion_KL(output.softmax(dim=-1).log(), batch_label.softmax(dim=-1))
            loss = MSE + lambda_KL * KL
            ind += 1

            Total_loss += loss.detach()
            MSE_LOSS += MSE.detach()
            KL_LOSS += KL.detach()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            output, _ = net(batch_data, 0.25)
            output = output.detach()

            d_real = dnet(batch_label)
            d_fake = dnet(output)

            real_label = Variable(torch.ones(d_real.size())).cuda()
            fake_label = Variable(torch.zeros(d_fake.size())).cuda()

            d_loss = criterion_BCE(d_fake, fake_label) + criterion_BCE(d_real, real_label)
            D_loss += d_loss.detach()
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()


        print("Epoch [",epoch,"] Loss:", Total_loss, " MSE: ", MSE_LOSS, " KL: ", KL_LOSS, "D_Loss: ", D_loss)
        
            

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if epoch % 200 == 0:
                torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.path')
                logging.info(f'Checkpoint {epoch + 1} saved')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = 15254

    net = DropNet(input_dim)
    dnet = Discriminator(input_dim)
    net.to(device=device)
    net.cuda()
    dnet.to(device=device)
    dnet.cuda()

    #net = nn.DataParallel(net, device_ids=[0,1])
    #dnet = nn.DataParallel(dnet, device_ids=[0,1])
    train_net(
        net,
        dnet,
        epochs=2001,
        batch_size=256,
        device=device,
        save_cp=True
    )
