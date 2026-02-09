import os, pickle
import numpy 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nutils
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import dataset.sigtest_dataset as dataset
from models.T_mamba import T_Mamba as Model
from models.dist import dist_seq, dist_seq_rf

parser = argparse.ArgumentParser(description='Online signature verification')
parser.add_argument('--train-shot-g', type=int, default=4, metavar='TRSG', 
                    help='number of genuine samples per class per training batch(default: 4)') #Genuine samples only
parser.add_argument('--train-shot-f', type=int, default=0, metavar='TRSG', 
                    help='number of forgery samples per class per training batch(default: 3)')
parser.add_argument('--train-tasks', type=int, default=1, 
                    help='number of tasks per batch')
parser.add_argument('--seed', type=int, default=111, metavar='S',
                    help='numpy random seed (default: 111)')
parser.add_argument('--epoch', type=str, default="End", 
                    help='model from the i-th epoch for testing')#"End"
parser.add_argument('--rf', action='store_true',
                    help='test random forgery or not')
parser.add_argument('--lr', type=float, default=0.001, 
                    help='learning rate')

parser.add_argument('--wisize', type=int, default=10, help='Window size for processing')
parser.add_argument('--stride', type=int, default=1, help='Stride for the window')
parser.add_argument('--sig_dep', type=int, default=2, help='Depth of the signature model')
parser.add_argument('--n_hidden', type=int, default=128, help='Number of hidden units in the model')
parser.add_argument('--d_state', type=int, default=256, help='State dimension')
parser.add_argument('--d_conv', type=int, default=4, help='Convolution dimension')
parser.add_argument('--expand', type=int, default=1, help='Expansion factor')
parser.add_argument('--n_out', type=int, default=64, help='Output dimension')
parser.add_argument('--gamma', type=int, default=5.0, help='SOFT-dtwCANSU gamma parameter')
parser.add_argument('--DTW', type=str, default="DTW",help="Choose which DTW implementation to use")

args = parser.parse_args()


modelname = 'T-Mamba'

n_task = args.train_tasks
n_shot_g = args.train_shot_g
n_shot_f = args.train_shot_f
print("Random Forgery Scenario:",args.rf)

numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

sigDict = pickle.load(open("/DeepSignDB/MCYT_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 25; num_f = 25
print("For MCYT:")
dataset_name = "MCYT" 
model_dataset_name = "%s_%s" % (args.modelname, dataset_name)

window_size=args.wisize
stride=args.stride
signature_depth = args.sig_dep

dset = dataset.dataset(sigDict=sigDict, finger_scene=False,window_size=11,stride=stride,signature_depth=signature_depth)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)
save_dir = "models/%d/%s" % (args.seed, modelname)

n_hidden=args.n_hidden
d_state=args.d_state
d_conv=args.d_conv
expand=args.expand
n_out=args.n_out
model = Model(
          n_in=dset.featDim, 
            n_hidden=n_hidden, 
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_out=n_out,
          n_task=n_task,
          n_shot_g=n_shot_g, 
          n_shot_f=n_shot_f, 
          initial_gamma = args.gamma,
          batchsize=num_g+num_f)
model.load_state_dict(torch.load("%s/epochEnd" % save_dir))



model.cuda()
model.train(mode=False)
model.eval()

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length= model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

save_path = "log/seed%d/%s" % (args.seed, model_dataset_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)

numpy.save("%s/dtw_dist_p%s.npy" % (save_path, args.epoch), DIST_P)
numpy.save("%s/dtw_dist_n%s.npy" % (save_path, args.epoch), DIST_N)
numpy.save("%s/dtw_dist_temp%s.npy" % (save_path, args.epoch), DIST_TEMP)


dataset_name = "BSID"
sigDict = pickle.load(open("/DeepSignDB/BSID_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 16; num_f = 12
print("For BiosecurID:")
model_dataset_name = "%s_%s" % (args.modelname, dataset_name)

dset = dataset.dataset(sigDict=sigDict, finger_scene=False,window_size=13,stride=stride,signature_depth=signature_depth)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

model.h0 = Variable(torch.zeros(2, num_g+num_f, 128).cuda(), requires_grad=False)

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

save_path = "log/seed%d/%s" % (args.seed, model_dataset_name)


if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("%s/dtw_dist_p%s.npy" % (save_path, args.epoch), DIST_P)
numpy.save("%s/dtw_dist_n%s.npy" % (save_path, args.epoch), DIST_N)
numpy.save("%s/dtw_dist_temp%s.npy" % (save_path, args.epoch), DIST_TEMP)

dataset_name = "BSDS2" 
model_dataset_name = "%s_%s" % (args.modelname, dataset_name)
sigDict = pickle.load(open("/DeepSignDB/BSDS2_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 19; num_f = 20
print("For Biosecure DS2:")

dset = dataset.dataset(sigDict=sigDict, finger_scene=False,window_size=5,stride=stride,signature_depth=signature_depth)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

model.h0 = Variable(torch.zeros(2, num_g+num_f, 128).cuda(), requires_grad=False)

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    idxs = numpy.concatenate([numpy.array([0,1,2,3]), numpy.arange(15, 50)])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

save_path = "log/seed%d/%s" % (args.seed, model_dataset_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("%s/dtw_dist_p%s.npy" % (save_path, args.epoch), DIST_P)
numpy.save("%s/dtw_dist_n%s.npy" % (save_path, args.epoch), DIST_N)
numpy.save("%s/dtw_dist_temp%s.npy" % (save_path, args.epoch), DIST_TEMP)


dataset_name = "EBio2"
model_dataset_name = "%s_%s" % (args.modelname, dataset_name)
sigDict = pickle.load(open("/DeepSignDB/EBio2_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 8; num_f = 6
print("For eBS DS2 w2:")

dset = dataset.dataset(sigDict=sigDict, finger_scene=False,window_size=11,stride=stride,signature_depth=signature_depth)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

model.h0 = Variable(torch.zeros(2, num_g+num_f, 128).cuda(), requires_grad=False)

feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio2
    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

save_path = "log/seed%d/%s" % (args.seed, model_dataset_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)
numpy.save("%s/dtw_dist_p%s.npy" % (save_path, args.epoch), DIST_P)
numpy.save("%s/dtw_dist_n%s.npy" % (save_path, args.epoch), DIST_N)
numpy.save("%s/dtw_dist_temp%s.npy" % (save_path, args.epoch), DIST_TEMP)

dataset_name = "EBio1" 
model_dataset_name = "%s_%s" % (args.modelname, dataset_name)
sigDict = pickle.load(open("/DeepSignDB/EBio1_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 8; num_f = 6

dset = dataset.dataset(sigDict=sigDict, finger_scene=False,window_size=window_size,stride=stride,signature_depth=signature_depth)
sampler = dataset.batchSampler(dset)
dataLoader = DataLoader(dset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

model.h0 = Variable(torch.zeros(2, num_g+num_f, 128).cuda(), requires_grad=False)

print("For eBS DS1 w1:")
feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio1
    device = 0
    idxs = numpy.concatenate([numpy.array([0,1,10,11,20,21,30,31]) + device * 2,
                              numpy.array([40,41,42,55,56,57]) + device * 3])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]
    sig = sig[:,0:int(numpy.max(lens)),:]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

save_path = "log/seed%d/%s" % (args.seed, model_dataset_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)

numpy.save("%s/dtw_dist_p%s_d0.npy" % (save_path, args.epoch), DIST_P)
numpy.save("%s/dtw_dist_n%s_d0.npy" % (save_path, args.epoch), DIST_N)
numpy.save("%s/dtw_dist_temp%s_d0.npy" % (save_path, args.epoch), DIST_TEMP)


print("For eBS DS1 w2:")
feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio1
    device = 1
    idxs = numpy.concatenate([numpy.array([0,1,10,11,20,21,30,31]) + device * 2,
                              numpy.array([40,41,42,55,56,57]) + device * 3])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]
    sig = sig[:,0:int(numpy.max(lens)),:]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

save_path = "log/seed%d/%s" % (args.seed, model_dataset_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)

numpy.save("%s/dtw_dist_p%s_d1.npy" % (save_path, args.epoch), DIST_P)
numpy.save("%s/dtw_dist_n%s_d1.npy" % (save_path, args.epoch), DIST_N)
numpy.save("%s/dtw_dist_temp%s_d1.npy" % (save_path, args.epoch), DIST_TEMP)

print("For eBS DS1 w3:")
feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio1
    device = 2
    idxs = numpy.concatenate([numpy.array([0,1,10,11,20,21,30,31]) + device * 2,
                              numpy.array([40,41,42,55,56,57]) + device * 3])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]
    sig = sig[:,0:int(numpy.max(lens)),:]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)
save_path = "log/seed%d/%s" % (args.seed, model_dataset_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)


if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)

numpy.save("%s/dtw_dist_p%s_d2.npy" % (save_path, args.epoch), DIST_P)
numpy.save("%s/dtw_dist_n%s_d2.npy" % (save_path, args.epoch), DIST_N)
numpy.save("%s/dtw_dist_temp%s_d2.npy" % (save_path, args.epoch), DIST_TEMP)

print("For eBS DS1 w4:")
feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio1
    device = 3
    idxs = numpy.concatenate([numpy.array([0,1,10,11,20,21,30,31]) + device * 2,
                              numpy.array([40,41,42,55,56,57]) + device * 3])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]
    sig = sig[:,0:int(numpy.max(lens)),:]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)
save_path = "log/seed%d/%s" % (args.seed, model_dataset_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)

numpy.save("%s/dtw_dist_p%s_d3.npy" % (save_path, args.epoch), DIST_P)
numpy.save("%s/dtw_dist_n%s_d3.npy" % (save_path, args.epoch), DIST_N)
numpy.save("%s/dtw_dist_temp%s_d3.npy" % (save_path, args.epoch), DIST_TEMP)

print("For eBS DS1 w5:")
feats = []
for idx, batch in enumerate(dataLoader):
    sig, lens, label = batch

    # For EBio1
    device = 4
    idxs = numpy.concatenate([numpy.array([0,1,10,11,20,21,30,31]) + device * 2,
                              numpy.array([40,41,42,55,56,57]) + device * 3])
    sig = sig[idxs]; lens = lens[idxs]; label = label[idxs]
    sig = sig[:,0:int(numpy.max(lens)),:]

    mask = model.getOutputMask(lens)

    sig = Variable(torch.from_numpy(sig)).cuda()
    mask = Variable(torch.from_numpy(mask)).cuda()
    label = Variable(torch.from_numpy(label)).cuda()

    output, length = model(sig, mask) #(N,T,D)
    output = output.data.cpu().numpy()
    feats.append(output)

save_path = "log/seed%d/%s" % (args.seed, model_dataset_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.rf:
    DIST_P, DIST_N, DIST_TEMP = dist_seq_rf(feats, n_shot_g, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP = dist_seq(feats, n_shot_g, 0, num_g, num_f)

numpy.save("%s/dtw_dist_p%s_d4.npy" % (save_path, args.epoch), DIST_P)
numpy.save("%s/dtw_dist_n%s_d4.npy" % (save_path, args.epoch), DIST_N)
numpy.save("%s/dtw_dist_temp%s_d4.npy" % (save_path, args.epoch), DIST_TEMP)

