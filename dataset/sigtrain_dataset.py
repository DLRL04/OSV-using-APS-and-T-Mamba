# -*- coding:utf-8 -*-
import numpy
import pickle 
from matplotlib import pyplot as plt
import sys
from dataset.utils import *

class dataset(object):
    """docstring for dataset"""
    def __init__(self, sigDict, taskSize=1, taskNumGen=5, taskNumNeg=5, saveKey=True, finger_scene=False,window_size=10,stride=1,signature_depth=2):
        super(dataset, self).__init__()
        self.trainKeys = list(sigDict.keys())
        self.taskSize = taskSize   #每个任务的数据大小
        self.taskNumGen = taskNumGen #每个用户选择的 genuine 数量
        self.taskNumNeg = taskNumNeg
        self.window_size = window_size
        self.stride= stride
        self.signature_depth =signature_depth
        self.feats = []
        self.numGen = numpy.zeros(len(self.trainKeys), dtype=numpy.int32)# 记录每个用户的样本数量；
        self.numNeg = numpy.zeros(len(self.trainKeys), dtype=numpy.int32)
        # The dataset is user-by-user arranged in the following format: anchors, genuines, forgeries, anchors, ...
        print (">>>>> Extracting features... <<<<<")
        for idx, key in enumerate(self.trainKeys):#通过 key 获取到的用户数据
            sys.stdout.write(">>>>> User key: %d <<<<<\r"%key)
            sys.stdout.flush()
            sigfeatExt(sigDict[key][True], self.feats, finger_scene=finger_scene,window_size=self.window_size,stride=self.stride,signature_depth=self.signature_depth)
            sigfeatExt(sigDict[key][False], self.feats, finger_scene=finger_scene,window_size=self.window_size,stride=self.stride,signature_depth=self.signature_depth)
            self.numGen[idx] = len(sigDict[key][True])# 记录每个用户的正负样本数量
            self.numNeg[idx] = len(sigDict[key][False])
        print (">>>>> Done <<<<<")
        self.accumNum2 = numpy.cumsum(self.numGen + self.numNeg) #得到每个用户数据的“结束位置”
        self.accumNum = numpy.roll(self.accumNum2, 1); self.accumNum[0] = 0  #开始索引
        
        self.featDim = self.feats[0].shape[1] #特征维度
        self.lens = numpy.zeros(len(self.feats), dtype=numpy.float32) #每个序列的时间长度
        for i, f in enumerate(self.feats):
            self.lens[i] = f.shape[0]

    def __getitem__(self, index):
        sig = self.feats[index]
        sigLen = self.lens[index]# 每个用户的实际长度
        sigLabel = numpy.sum(index>=self.accumNum2) # Note that keys start from 1, while labels start from 0. 
        # 哪个用户（用于监督学习标签）
        # sigLabel = int(self.label_is_genuine[index])  # 1 表示真，0 表示伪

        return sig, sigLen, sigLabel

    def __len__(self):
        return len(self.trainKeys) 

    def addDatabase(self, sigDict, finger_scene=False, window_size=None, stride=None, signature_depth=None):
        newKeys = list(sigDict.keys())
        self.trainKeys = numpy.concatenate((self.trainKeys, newKeys))
        N = len(self.feats)

        numGen = numpy.zeros(len(newKeys), dtype=numpy.int32)
        numNeg = numpy.zeros(len(newKeys), dtype=numpy.int32)
        print(">>>>> addDatabase——Extracting features... <<<<<")

        # 使用新的参数，如果未指定则用当前对象的默认值
        window_size = window_size or self.window_size
        stride = stride or self.stride
        signature_depth = signature_depth or self.signature_depth

        for i, key in enumerate(newKeys):
            sys.stdout.write(f">>>>> User key: {key} <<<<<\r")
            sys.stdout.flush()

            sigfeatExt(sigDict[key][True], self.feats,
                    finger_scene=finger_scene,
                    window_size=window_size,
                    stride=stride,
                    signature_depth=signature_depth)
            
            sigfeatExt(sigDict[key][False], self.feats,
                    finger_scene=finger_scene,
                    window_size=window_size,
                    stride=stride,
                    signature_depth=signature_depth)

            numGen[i] = len(sigDict[key][True])
            numNeg[i] = len(sigDict[key][False])

        print(">>>>> Done <<<<<")
        self.numGen = numpy.concatenate((self.numGen, numGen))
        self.numNeg = numpy.concatenate((self.numNeg, numNeg))
        self.accumNum2 = numpy.cumsum(self.numGen + self.numNeg)
        self.accumNum = numpy.roll(self.accumNum2, 1)
        self.accumNum[0] = 0

        lens = numpy.zeros(len(self.feats) - N, dtype=numpy.float32)
        for i in range(N, len(self.feats)):
            lens[i - N] = self.feats[i].shape[0]
        self.lens = numpy.concatenate((self.lens, lens))


    # def histLens(self):
    #     lens = numpy.zeros(len(self.feats))
    #     for idx, f in enumerate(self.feats):
    #         lens[idx] = f.shape[0]
    #     plt.hist(lens, bins=30)
    #     plt.show()
        
class batchSampler(object):
    """docstring for sampler"""
    def __init__(self, dataset, loop=False):
        super(batchSampler, self).__init__()
        self.taskSize = dataset.taskSize
        self.index = numpy.arange(0, len(dataset.trainKeys), dtype=numpy.int32)
        # self.index = numpy.repeat(self.index, self.taskSize, axis=0)
        self.taskNumGen = dataset.taskNumGen
        self.taskNumNeg = dataset.taskNumNeg
        self.numGen = dataset.numGen
        self.numNeg = dataset.numNeg
        self.accumNum = dataset.accumNum
        self.numIters = len(dataset)
        self.loop = loop

    def __iter__(self):
        batch = []
        numpy.random.shuffle(self.index)
        for i in range(self.numIters):# 每批任务包括 taskSize 个用户
            if self.loop:
                idxs = self.index[numpy.arange(i, i+self.taskSize)%len(self.index)]
            else:
                idxs = numpy.random.choice(self.index, size=self.taskSize, replace=False)
            for idx in idxs:
                gen = numpy.random.choice(self.numGen[idx], size=1+self.taskNumGen, replace=False) 
                ## SF
                # neg = numpy.random.choice(self.numNeg[idx], size=self.taskNumNeg, replace=False) + self.numGen[idx]
                # batch.append(gen + self.accumNum[idx])
                # batch.append(neg + self.accumNum[idx])
                ## SF + RF
                neg = numpy.random.choice(self.numNeg[idx], size=self.taskNumNeg//2, replace=False) + self.numGen[idx]
                batch.append(gen + self.accumNum[idx])
                batch.append(neg + self.accumNum[idx])
                idxs_RF = (idx+numpy.random.randint(1, len(self.index), size=self.taskNumNeg//2))%len(self.index)
                for idx in idxs_RF:
                    neg = numpy.random.choice(self.numGen[idx], size=1, replace=False) 
                    batch.append(neg + self.accumNum[idx])
                ## RF
                # batch.append(gen + self.accumNum[idx])
                # idxs_RF = (idx+numpy.random.randint(1, len(self.index), size=self.taskNumNeg))%len(self.index)
                # for idx in idxs_RF:
                #     neg = numpy.random.choice(self.numGen[idx], size=1, replace=False) 
                #     batch.append(neg + self.accumNum[idx])
            batch = numpy.concatenate(batch, axis=0).astype(numpy.int32)
            yield batch
            batch = []

    def __len__(self):
        return self.numIters

def collate_fn(batch):
    ''' `batch` is a list of tuple where 1-st element is the signature, 2-nd element is its length and 3-rd element is the label.
    '''
    batchSize = len(batch)
    sig = [item[0] for item in batch]
    sigLen = numpy.array([item[1] for item in batch], dtype=numpy.float32)
    sigLabel = numpy.array([item[2] for item in batch], dtype=numpy.int64)
    maxLen = int(numpy.max(sigLen))

    # print (sigLabel)
    # plt.plot(sig[0][:, -2], sig[0][:, -1])
    # for i in range(1, 6):
    #     plt.plot(sig[i][:, -2] + (i - 1) * 1.0, sig[i][:, -1] + 1.0)
    # for i in range(6, 11):
    #     plt.plot(sig[i][:, -2] + (i - 6) * 1.0, sig[i][:, -1] + 2.0)
    # plt.show()

    sigPadded = numpy.zeros((batchSize, maxLen, sig[0].shape[1]), dtype=numpy.float32)
    for idx, s in enumerate(sig):
        sigPadded[idx,:s.shape[0]] = s

    return sigPadded, sigLen, sigLabel
