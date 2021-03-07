import matplotlib.pyplot as plt
import argparse as ae

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

def doublemoon_sample(n_sample):
    x1_1 = Normal(4, 4)
    sampled_x1_1 = x1_1.sample((int(n_sample/2),))
    x2_1 = Normal(0.25*(sampled_x1_1-4)**2-20, torch.ones_like(sampled_x1_1)*2)
    sampled_x2_1 = x2_1.sample()
        
    x1_2 = Normal(-4, 4)
    sampled_x1_2 = x1_2.sample((int(n_sample/2),))
    x2_2 = Normal(-0.25*(sampled_x1_2+4)**2+20, torch.ones_like(sampled_x1_2)*2)
    sampled_x2_2 = x2_2.sample()
        
    sampled_x1 = torch.cat([sampled_x1_1, sampled_x1_2])
    sampled_x2 = torch.cat([sampled_x2_1, sampled_x2_2])
    sampled_x = torch.zeros(n_sample, 2)
    sampled_x[:,0] = sampled_x1*0.2
    sampled_x[:,1] = sampled_x2*0.1
    
    return sampled_x
    
class NN(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3_s = nn.Linear(n_hidden, n_input)
        self.fc3_t = nn.Linear(n_hidden, n_input)

    def forward(self, x):
        hidden = F.relu(self.fc2(F.relu(self.fc1(x))))
        s = torch.tanh(self.fc3_s(hidden))
        t = self.fc3_t(hidden)
        return s, t

class RealNVP(nn.Module):
    def __init__(self, n_flows, data_dim, n_hidden,device="cpu"):
        super(RealNVP, self).__init__()
        self.n_flows = n_flows
        self.NN = torch.nn.ModuleList()
        self.device = device
        
        assert(data_dim % 2 == 0)
        self.n_half = int(data_dim/2)
            
        for k in range(n_flows):
            self.NN.append(NN(self.n_half, n_hidden))
            
        self.NN.to(device)
                
    def forward(self, x, n_layers=None):
        if n_layers == None:
            n_layers = self.n_flows
                
        log_det_jacobian = 0
        for k in range(n_layers):
            x_a = x[:, :self.n_half]
            x_b = x[:, self.n_half:]
            s, t = self.NN[k](x_a)
            x_b = torch.exp(s)*x_b + t
            x = torch.cat([x_b, x_a], dim=1)
            log_det_jacobian += s
        return x, log_det_jacobian
                        
    def inverse(self, z, n_layers=None):
        if n_layers == None:
            n_layers = self.n_flows
        for k in reversed(range(n_layers)):
            z_a = z[:, self.n_half:]
            z_b = z[:, :self.n_half]
            s, t = self.NN[k](z_a)
            z_b = (z_b - t) / torch.exp(s)
            z = torch.cat([z_a, z_b], dim=1)
        return z

def train(model,n_epochs,train_loader,optimizer,z):
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0
        for sample_x in train_loader:
            sample = sample_x.to(device)
            optimizer.zero_grad()
            sample_z, log_det_jacobian = model(sample)
            loss = -1 * torch.mean(z.log_prob(sample_z) + log_det_jacobian)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('epoch:[%2d/%2d] loss:%1.4f' % (epoch+1, n_epochs, running_loss / len(train_loader)))
        
# main
parser = ae.ArgumentParser()
parser.add_argument("-igpu","--igpu",required=True,help="input gpu label.")
parser.add_argument("-ns","--n_sample",default=10000,help="number of samles.")
parser.add_argument("-bs","--batch_size",default=64,help="batch size.")
parser.add_argument("-ne","--nepoch",type=int,default=50,help="number of epithod.")
args = parser.parse_args()
device = torch.device("cuda:"+str(args.igpu))

z = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
sampled_z = z.sample((args.n_sample,))
plt.figure(figsize = (5,5))
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.scatter(sampled_z[:,0].cpu(), sampled_z[:,1].cpu(), s=15)

sampled_x = doublemoon_sample(args.n_sample)
plt.figure(figsize = (5,5))
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.scatter(sampled_x[:,0].cpu(),sampled_x[:,1].cpu(), s=15)

# train
train_loader = DataLoader(sampled_x, batch_size=args.batch_size, shuffle=True)
model = RealNVP(4, 2, 256).to(device)
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train(model,args.nepoch,train_loader,optimizer,z)

#predict
test_z = z.sample((args.n_sample,))
test_x= model.inverse(test_z)
test=test_x.cpu().detach().numpy()
plt.figure(figsize = (5,5))
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.scatter(test[:,0],test[:,1], s=15)

plt.show()

