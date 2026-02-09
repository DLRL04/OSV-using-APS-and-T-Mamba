# -*- coding: UTF-8 -*-

import pickle
import numpy 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import dataset.sigtrain_dataset as dataset
import matplotlib.pyplot as plt

from models.T_mamba import T_Mamba as Model

parser = argparse.ArgumentParser(description='Dynamic signature verification')

parser.add_argument('--train-shot-g', type=int, default=5, metavar='TRSG', 
                    help='number of genuine samples per class per training batch(default: 5)') 
parser.add_argument('--train-shot-f', type=int, default=10, metavar='TRSG',
                    help='number of forgery samples per class per training batch(default: 10)')
parser.add_argument('--train-tasks', type=int, default=4, 
                    help='number of tasks per batch')
parser.add_argument('--epochs', type=int, default=20, 
                    help='number of epochs to train (default: 40)')
parser.add_argument('--seed', type=int, default=222, metavar='S',
                    help='numpy random seed (default: 222)')
parser.add_argument('--save-interval', type=int, default=3, 
                    help='how many epochs to wait before saving the model.')
parser.add_argument('--lr', type=float, default=0.001, 
                    help='learning rate')

parser.add_argument('--stride', type=int, default=1, help='Stride for the window')
parser.add_argument('--sig_dep', type=int, default=2, help='Depth of the signature model')
parser.add_argument('--n_hidden', type=int, default=128, help='Number of hidden units in the model')
parser.add_argument('--d_state', type=int, default=256, help='State dimension')
parser.add_argument('--d_conv', type=int, default=4, help='Convolution dimension')
parser.add_argument('--expand', type=int, default=1, help='Expansion factor')
parser.add_argument('--gamma', type=int, default=5.0, help='SOFT-dtwCANSU gamma parameter')

args = parser.parse_args()

n_task = args.train_tasks
n_shot_g = args.train_shot_g
n_shot_f = args.train_shot_f

numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True
modelname = 'T-Mamba'
sigDict = pickle.load(open("/DeepSignDB/MCYT_dev.pkl", "rb"), encoding='iso-8859-1')
stride=args.stride
signature_depth = args.sig_dep
dset = dataset.dataset(
                    sigDict=sigDict,
                    taskSize=n_task, 
                    taskNumGen=n_shot_g, 
                    taskNumNeg=n_shot_f,
                    finger_scene=False,                    
                    window_size=11,
                    stride=stride,
                    signature_depth=signature_depth
                )

del sigDict
sigDict = pickle.load(open("/DeepSignDB/BSID_dev.pkl", "rb"), encoding='iso-8859-1')
dset.addDatabase(sigDict, finger_scene=False, window_size=13, stride=stride, signature_depth=signature_depth)
del sigDict
sigDict = pickle.load(open("/DeepSignDB/EBio1_dev.pkl", "rb"), encoding='iso-8859-1')
dset.addDatabase(sigDict, finger_scene=False, window_size=5, stride=stride, signature_depth=signature_depth)
del sigDict
sigDict = pickle.load(open("/DeepSignDB/EBio2_dev.pkl", "rb"), encoding='iso-8859-1')
dset.addDatabase(sigDict, finger_scene=False, window_size=11, stride=stride, signature_depth=signature_depth)
del sigDict

sampler = dataset.batchSampler(dset, loop=False)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)
n_hidden=args.n_hidden
d_state=args.d_state
d_conv=args.d_conv
expand=args.expand
model = Model(
            n_in=dset.featDim, 
             n_hidden=n_hidden, 
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_task=n_task,
            n_shot_g=n_shot_g,
            n_shot_f=n_shot_f
        )
model.train(mode=True)
model.cuda()


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) 
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 
save_dir = "models/%d/%s" % (args.seed, modelname)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
loss_std_list = []
loss_hard_list = []
var_list = []
epochs_list = []
for epoch in range(0, args.epochs):
    TriLoss_std = 0
    TriLoss_hard = 0
    Var = 0

    for idx, batch in enumerate(dataLoader):
        sig, lens, label = batch
        mask = model.getOutputMask(lens)

        sig = Variable(torch.from_numpy(sig)).cuda()
        mask = Variable(torch.from_numpy(mask)).cuda()
        label = Variable(torch.from_numpy(label)).cuda()

        optimizer.zero_grad()
        output, length= model(sig, mask) #(N,T,D)

        triLoss_std, triLoss_hard, var = model.tripletLoss(output, length)
        (triLoss_hard+0.01*var).backward() 
        optimizer.step()
        
        TriLoss_std += triLoss_std.item()
        TriLoss_hard += triLoss_hard.item()
        Var += var.item()

        if (idx + 1) % 20 == 0:
            print ("epoch:",epoch, "idx:",idx)
            print("TriLoss_std:", format(TriLoss_std/50,'.6f'), "TriLoss_hard:", \
                format(TriLoss_hard/50,'.6f'), "Var:", format(Var/50,'.6f'))
            TriLoss_std = 0
            TriLoss_hard = 0
            Var = 0
    loss_std_list.append(TriLoss_std / len(dataLoader))
    loss_hard_list.append(TriLoss_hard / len(dataLoader))
    var_list.append(Var / len(dataLoader))
    epochs_list.append(epoch)

    lr_scheduler.step()
    
    if epoch % args.save_interval == 0:
       model_path = "%s/epoch%d" % (save_dir, epoch)
       torch.save(model.state_dict(), model_path)
best_model_path = "%s/epochEnd" % save_dir
if os.path.exists(best_model_path):
    os.remove(best_model_path)
print(f"✅ [INFO] Model saved to {best_model_path}")

torch.save(model.state_dict(), best_model_path)

