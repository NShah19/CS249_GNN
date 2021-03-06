import os
import numpy as np
import argparse
import time
import copy

import matplotlib.pyplot as plt
import deepdish as dd

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from AbideData import AbideDataset
from torch_geometric.data import DataLoader
from net.brain_networks import NNGAT_Net

from utils.utils import normal_transform_train
from utils.mmd_loss import MMD_loss
import sklearn

import optuna
from optuna.trial import TrialState

torch.manual_seed(123)

EPS = 1e-15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=100, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='ABIDE_data', help='root directory of the dataset')
parser.add_argument('--dataset', type=str, default='abide', help='abide || hcp')
parser.add_argument('--fold', type=int, default=1, help='training which fold')
parser.add_argument('--lr', type = float, default=0.01, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=1, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=1, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0.1, help='s1 distance regularization')
parser.add_argument('--lamb4', type=float, default=0.1, help='s2 distance regularization')
parser.add_argument('--lamb5', type=float, default=0, help='s1 consistence regularization')
parser.add_argument('--lamb6', type=float, default=0, help='s2 consistence regularization')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--normalization', action='store_true')
parser.set_defaults(save_model=True)
parser.set_defaults(normalization=True)
opt = parser.parse_args()



############# Define Dataloader -- need costumize#####################
if opt.dataset == 'hcp':
    # TODO: Load HCP Data
    pass
else:
    dataset = AbideDataset(opt.dataroot, 'ABIDE')
    indim = 116
    nclass = 2

############### split train, val, and test set -- need costumize########################
n_train = int(len(dataset)*.6)
n_val = int(len(dataset)*.2)
n_test = len(dataset) - n_train - n_val

tr_index, val_index, te_index = torch.utils.data.random_split(list(range(len(dataset))), (n_train, n_val, n_test))

train_mask = torch.zeros(len(dataset), dtype=torch.uint8)
test_mask = torch.zeros(len(dataset), dtype=torch.uint8)
val_mask = torch.zeros(len(dataset), dtype=torch.uint8)
train_mask[tr_index] = 1
test_mask[te_index] = 1
val_mask[val_index] = 1
test_dataset = dataset[test_mask]
train_dataset = dataset[train_mask]
val_dataset = dataset[val_mask]


# ######################## Data Preprocessing ########################
# ###################### Normalize features ##########################
if opt.normalization:
    for i in range(train_dataset.data.x.shape[1]):
        train_dataset.data.x[:, i], lamb, xmean, xstd = normal_transform_train(train_dataset.data.x[:, i])

test_loader = DataLoader(test_dataset,batch_size=opt.batchSize,shuffle = False)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)



############################### Define Other Loss Functions ########################################

def dist_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res

def consist_loss(s):
    if len(s) == 0:
        return 0
    else:
        s = torch.sigmoid(s)
        W = torch.ones(s.shape[0],s.shape[0])
        D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
        L = D-W
        L = L.to(device)
        res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
        return res

###################### Network Training Function#####################################
def train(model, optimizer, scheduler, epoch, lamb0, lamb3, lamb4, lamb5, ratio):
    print('train...........')
    model.train()

    s1_list = []
    s2_list = []
    loss_all = 0
    loss_en1_all  = 0
    loss_en2_all = 0

    i = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr)

        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())

        loss_c = F.nll_loss(output, data.y) # classification loss

        loss_dist1 = dist_loss(s1, ratio)
        loss_dist2 = dist_loss(s2, ratio)
        loss_consist = consist_loss(s1[data.y == 1]) + consist_loss(s1[data.y == 0])
        loss = lamb0 * loss_c + lamb3 * loss_dist1 + lamb4 * loss_dist2 + lamb5 * loss_consist

        i = i + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        loss_en1_all +=loss_dist1.item() *data.num_graphs
        loss_en2_all += loss_dist2.item() * data.num_graphs
        optimizer.step()
        scheduler.step()

        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)

    return loss_all / len(train_dataset), s1_arr, s2_arr, loss_en1_all / len(train_dataset),loss_en2_all / len(train_dataset)


###################### Network Testing Function#####################################
def test_acc(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output,_,_= model(data.x, data.edge_index, data.batch, data.edge_attr)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def test_loss(model, loader, epoch, lamb0, lamb3, lamb4, lamb5, ratio):
    print('testing...........')
    model.eval()
    loss_all = 0

    i=0
    for data in loader:
        data = data.to(device)
        output,s1,s2 = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss_c = F.nll_loss(output, data.y)

        loss_dist1 = dist_loss(s1, ratio)
        loss_dist2 = dist_loss(s2, ratio)
        loss_consist = consist_loss(s1)
        loss = lamb0 * loss_c + lamb3 * loss_dist1 + lamb4 * loss_dist2 + lamb5 * loss_consist

        i = i + 1

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

#######################################################################################
############################   Model Training #########################################
#######################################################################################
############### Define Graph Deep Learning Network ##########################
def objective(trial):
    # Generate the model.
    pool_method = 'topk'
    ratio = .5
    model = NNGAT_Net(ratio, indim=indim).to(device)

    print(model)
    print('ground_truth: ', test_dataset.data.y, 'total: ', len(test_dataset.data.y), 'positive: ',sum(test_dataset.data.y))

    # Generate the optimizers.
    optimizer_name = 'Adam'
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = .9
    weight_decay = .05
    if optimizer_name == 'Adam':
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=gamma)
    lamb0 = trial.suggest_float("lamb0", 0, 1)
    lamb3 = trial.suggest_float("lamb3", 0, 1)
    lamb4 = trial.suggest_float("lamb4", 0, 1)
    lamb5 = trial.suggest_float("lamb5", 0, 1)

    for epoch in range(0, opt.n_epochs):
        since  = time.time()
        tr_loss, s1_arr, s2_arr,le1,le2 = train(model, optimizer, scheduler, epoch, lamb0, lamb3, lamb4, lamb5, ratio)
        tr_acc = test_acc(model, train_loader)
        val_acc = test_acc(model, val_loader)
        val_loss = test_loss(model, val_loader, epoch, lamb0, lamb3, lamb4, lamb5, ratio)
        time_elapsed = time.time() - since
        print('*====**')
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Epoch: {:03d}, Train Loss: {:.7f}, '
            'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                        tr_acc, val_loss, val_acc))

        trial.report(val_acc, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc

#######################################################################################
######################### Create Optuna Study ######################################
#######################################################################################
if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize"])
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    fig1 = optuna.visualization.plot_parallel_coordinate(study)
    fig1.show()
    fig2 = optuna.visualization.plot_contour(study)
    fig2.show()
    fig3 = optuna.visualization.plot_slice(study)
    fig3.show()
    fig4 = optuna.visualization.plot_optimization_history(study)
    fig4.show()
    fig5 = optuna.visualization.plot_param_importances(study)
    fig5.show()
    
