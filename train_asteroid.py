import os
import argparse
import torch
import pickle
import torch.utils.data as utils
import torch.optim as optim
import time
import numpy as np

from graph import Graph
from model import GNet
from pool import FeaturePooling
from metrics import loss_function
from data_asteroid import CustomDatasetFolder

# Args
parser = argparse.ArgumentParser(description='Pixel2Mesh training script')
parser.add_argument('--data', type=str, default=None, metavar='D',
                    help="folder where data is located.")
parser.add_argument('--data_val', type=str, default=None, metavar='D',
                    help="folder where validation data is located.")
parser.add_argument('--epochs', type=int, default=200, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                    help='learning rate (default: 3e-5)')
parser.add_argument('--log_step', type=int, default=100, metavar='LS',
                    help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--saving_step', type=int, default=100, metavar='S',
                    help='how many batches to wait before saving model (default: 1000)')
parser.add_argument('--experiment', type=str, default='./model/', metavar='E',
                    help='folder where model and optimizer are saved.')
parser.add_argument('--load_model', type=str, default=None, metavar='M',
                    help='model file to load to continue training.')
parser.add_argument('--load_optimizer', type=str, default=None, metavar='O',
                    help='model file to load to continue training.')
args = parser.parse_args()

# Cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model
nIms = 25
if args.load_model is not None: # Continue training
    state_dict = torch.load(args.load_model, map_location=device)
    model_gcn = GNet(nIms)
    model_gcn.load_state_dict(state_dict)
else:
    model_gcn = GNet(nIms)

# Optimizer
if args.load_optimizer is not None:
    state_dict_opt = torch.load(args.load_optimizer, map_location=device)
    optimizer = optim.Adam(model_gcn.parameters(), lr=args.lr)
    optimizer.load_state_dict(state_dict_opt)
else:
    optimizer = optim.Adam(model_gcn.parameters(), lr=args.lr)

# Graph
graph = Graph("./ellipsoid/init_info.pickle")

# Data Loader
train_folder = CustomDatasetFolder(args.data, extensions = ["png"], dimension=nIms)
train_loader = torch.utils.data.DataLoader(train_folder, batch_size=1, shuffle=True)
# val_folder = CustomDatasetFolder(args.data_val, extensions = ["png"], dimension=nIms)
# val_loader = torch.utils.data.DataLoader(val_folder, batch_size=1, shuffle=True)

# Param
nb_epochs = args.epochs
log_step = args.log_step
saving_step = args.saving_step
curr_loss = 0

# To GPU
if use_cuda:
    print('Using GPU', flush=True)
    model_gcn.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
else:
    print('Using CPU', flush=True)

print("nb trainable param", model_gcn.get_nb_trainable_params(), flush=True)

model_gcn.train()
# Train
for epoch in range(1, nb_epochs+1):
    for n, data in enumerate(train_loader):
        ims, viewpoints, gt_points, gt_normals = data
        ims = np.transpose(ims, (1, 0, 2, 3, 4))
        viewpoints = np.transpose(viewpoints, (1, 0, 2))
        m, b, h, w, c = ims.shape

        t_ims = []
        for i in range(m):
            v = viewpoints[i].flatten()
            v = np.tile(v, (b, h, w, 1))
            im = np.concatenate((ims[i], v), axis=3)
            t_ims.append(im)
        t_ims = torch.from_numpy(np.asarray(t_ims))

        if use_cuda:
            t_ims = t_ims.cuda()
            gt_points = gt_points.cuda()
            gt_normals = gt_normals.cuda()

        # Forward
        graph.reset()
        optimizer.zero_grad()
        pools = []
        for i in range(m):
            pools.append(FeaturePooling(t_ims[i]))

        pred_points = model_gcn(graph, pools)

        # Loss
        loss = loss_function(pred_points, gt_points.squeeze(),
                                          gt_normals.squeeze(), graph)

        # Backward
        loss.backward()
        optimizer.step()

        curr_loss += loss

    # Log train
    print("Epoch", epoch, flush=True)
    print("Train Loss:", curr_loss.data.item()/len(train_loader.dataset), flush=True)
    curr_loss = 0

    # Save model
    if epoch % 10 == 0:
        model_file = args.experiment + "model_" + str(epoch) + ".pth"
        optimizer_file = args.experiment + "optimizer_" + str(epoch) + ".pth"
        torch.save(model_gcn.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        print("Saved model to " + model_file, flush=True)

    # model_gcn.eval()
    # for n, data in enumerate(val_loader):
    #     ims, gt_points, gt_normals = data
    #     ims = np.transpose(ims, (1, 0, 2, 3, 4))
    #     m, b, *x_dims = ims.shape

    #     if use_cuda:
    #         ims = ims.cuda()
    #         gt_points = gt_points.cuda()
    #         gt_normals = gt_normals.cuda()

    #     # Forward
    #     graph.reset()
    #     pools = []
    #     for i in range(m):
    #         pools.append(FeaturePooling(ims[i]))
    #     pred_points = model_gcn(graph, pools)

    #     # Loss
    #     loss = loss_function(pred_points, gt_points.squeeze(),
    #                                       gt_normals.squeeze(), graph)
    #     curr_loss += loss

    # # Log val
    # print("Epoch", epoch, flush=True)
    # print("Val Loss:", curr_loss.data.item()/len(val_loader.dataset), flush=True)
    # curr_loss = 0