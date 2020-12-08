import matplotlib.pyplot as plt
import pickle,os,time
import numpy as np
from torchvision import datasets,transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torch,torchvision
from AISEv2 import AISE
from mnist_model import CNN
from attack import PGD

ROOT = "datasets"
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.0,std=1.0)
])
N_TRAIN = 60000
N_EVAL = 1000


trainset = datasets.MNIST(root=ROOT,train=True,transform=TRANSFORM,download=False)
testset = datasets.MNIST(root=ROOT,train=False,transform=TRANSFORM,download=False)

np.random.seed(1234)
ind_train = np.arange(len(trainset))
np.random.shuffle(ind_train)
ind_train = ind_train[:N_TRAIN]

ind_eval = np.arange(len(testset))
np.random.shuffle(ind_eval)
ind_eval = ind_eval[200:200+N_EVAL]

x_train = trainset.data[ind_train].unsqueeze(1)/255.
y_train = trainset.targets[ind_train]
x_eval = testset.data[ind_eval].unsqueeze(1)/255.
y_eval = testset.targets[ind_eval]

DATA_TYPE_SHORT = "adv"
EPS = 40
DEVICE = torch.device("cuda")
net = CNN()
net.load_state_dict(torch.load("./models/mnistmodel.pt",map_location=DEVICE)["state_dict"])
net.eval()
for parameter in net.parameters():
    parameter.requires_grad_(False)
net.to(DEVICE)

fbcs = ["bcells_mnist_adv_conv{}_60000_1000_{}.pkl".format(i,EPS) for i in range(1,5)]
b_cells = []
b_labs = []
for fname in fbcs:
    with open("adaptive_results/"+fname,"rb") as f:
        mem_bcells,mem_labs = pickle.load(f)
        b_cells.append(mem_bcells.reshape((N_EVAL,20,)+mem_bcells.shape[2:])[:,:3])
        b_labs.append(mem_labs[:,-10:])

x_adv = PGD(eps=EPS/255.,sigma=20/255.,nb_iter=20,
    DEVICE=DEVICE).attack_batch(net,x_eval.to(DEVICE),y_eval.to(DEVICE))
with open("adaptive_cache/x_mnist_{}_{}v2_{}.pkl".format(DATA_TYPE_SHORT,N_EVAL,EPS), "rb") as f:
    x_adv = torch.Tensor(pickle.load(f))
#     pickle.dump(x_adv.detach().cpu().numpy(), f)

def feature_extractor(net,x,hidden_layer=-1,batch_size=2048,device=DEVICE):
    if hidden_layer == -1:  # return the last output of net
        outs = []
        for i in range(0,x.size(0),batch_size):
            xx = x[i:i+batch_size]
            *_, out = net(xx.to(device))
            outs.append(out.detach().cpu())
        return torch.cat(outs,dim=0)
    else:
        out_hiddens = []
        for i in range(0,x.size(0),batch_size):
            xx = x[i:i+batch_size]
            *out_hidden,_ = [h.detach().cpu() for h in net(xx.to(device))]
            out_hiddens.append(out_hidden[hidden_layer])
        return torch.cat(out_hiddens,dim=0)

out = feature_extractor(net,x_adv)
y_pred = torch.max(out, 1)[1]

print('The accuracy of plain cnn under PGD attacks is: {}'.format(
    (y_eval.numpy() == y_pred.detach().cpu().numpy()).astype("float").mean()))
   
for hidden_layer in range(4):
    sampling_temperature = [3.0,18.0,18.0,72.0][hidden_layer]
    x_train_hardened = torch.cat([x_train,torch.Tensor(b_cells[hidden_layer]).reshape(-1,1,28,28)])
    y_train_hardened = torch.cat([y_train,torch.LongTensor(b_labs[hidden_layer]).flatten()])
    aise = AISE(x_train_hardened,y_train_hardened,model=net,n_neighbors=10,hidden_layer=hidden_layer,sampling_temperature=sampling_temperature)
    mem_bcs, mem_labs, pla_bcs, pla_labs, ant_logs = aise(x_adv,y_eval)
    aise_proba = AISE.predict_proba(pla_labs.astype("int"),n_class=10)
    aise_pred = aise_proba.argmax(axis=1)
    aise_acc = (aise_pred==y_eval.numpy()).astype("float").mean()
    print("The accuracy by AISE on hardened Conv{} layer of adversarial examples is {}".format(hidden_layer+1,aise_acc))
    with open("adaptive_results/result_hardenv2_conv{}_1000_add3000_40.pkl".format(hidden_layer+1),"wb") as f:
        pickle.dump([aise_proba,ant_logs],f)