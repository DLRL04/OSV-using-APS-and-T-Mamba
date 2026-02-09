# -*- coding:utf-8 -*-
import numpy, pdb
import torch
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

import iisignature
from fastdtw import fastdtw as dtw #https://github.com/slaypni/fastdtw/issues

def diff(x):
    dx = numpy.convolve(x, [0.5,0,-0.5], mode='same'); dx[0] = dx[1]; dx[-1] = dx[-2]
    # dx = numpy.convolve(x, [0.2,0.1,0,-0.1,-0.2], mode='same'); dx[0] = dx[1] = dx[2]; dx[-1] = dx[-2] = dx[-3]
    return dx

def diffTheta(x):
    dx = numpy.zeros_like(x)
    dx[1:-1] = x[2:] - x[0:-2]; dx[-1] = dx[-2]; dx[0] = dx[1]
    temp = numpy.where(numpy.abs(dx)>numpy.pi)
    dx[temp] -= numpy.sign(dx[temp]) * 2 * numpy.pi
    dx *= 0.5
    return dx
#  巴特沃斯低通滤波器，用来平滑轨迹数据，去除高频噪声
class butterLPFilter(object):
    """docstring for butterLPFilter"""
    def __init__(self, highcut=10.0, fs=200.0, order=3):
        super(butterLPFilter, self).__init__()
        nyq = 0.5 * fs
        highcut = highcut / nyq
        b, a = signal.butter(order, highcut, btype='low')
        self.b = b
        self.a = a
    def __call__(self, data):
        y = signal.filtfilt(self.b, self.a, data) #双向滤波，能避免相位延迟，适合平滑轨迹数据
        return y

bf = butterLPFilter(15, 100)
def featExt(pathList, feats, gpnoise=None, dim=2, transform=False, finger_scene=False):
    for path in pathList:
        p = path[:,dim]
        path = path[:, 0:dim] #(x,y,p)
        path[:,0] = bf(path[:,0])
        path[:,1] = bf(path[:,1])
        
        dx = diff(path[:, 0]); dy = diff(path[:, 1])
        v = numpy.sqrt(dx**2+dy**2)
        theta = numpy.arctan2(dy, dx)
        cos = numpy.cos(theta)
        sin = numpy.sin(theta)
        dv = diff(v)
        dtheta = numpy.abs(diffTheta(theta))
        logCurRadius = numpy.log((v+0.05) / (dtheta+0.05))
        dv2 = numpy.abs(v*dtheta)
        totalAccel = numpy.sqrt(dv**2 + dv2**2)

        feat = numpy.concatenate((dx[:,None], dy[:,None], v[:,None], cos[:,None], sin[:,None], theta[:,None], 
                                  logCurRadius[:,None], totalAccel[:,None], dv[:,None], dv2[:,None], dtheta[:,None], p[:,None]), axis=1).astype(numpy.float32) #A minimum and well-performed feature set. 
        
        if finger_scene:
            ''' For finger scenario '''
            feat[:,:-1] = (feat[:,:-1] - numpy.mean(feat[:,:-1], axis=0)) / numpy.std(feat[:,:-1], axis=0)
        else:
            ''' For stylus scenario '''
            #print('feats',feat.shape,numpy.isnan(feat).any(),numpy.isinf(feat).any())都为FALSE
            feat = (feat - numpy.mean(feat, axis=0)) / numpy.std(feat, axis=0)
        #print('feat',feat.shape)就是不同的时间长度
        feats.append(feat.astype(numpy.float32))
    return feats
import numpy as np

def leadlag_transform(x):
    """
    严格 Lead–Lag 变换
    输入:
        x: numpy 数组, shape (T, D) 或 (T,)  一维或多维时间序列
    输出:
        ll_path: numpy 数组, shape (2*T - 2, 2*D)  Lead–Lag 轨迹
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]  # 转为二维 (T, 1)

    # 每个时间点重复两次
    x_rep = np.repeat(x, 2, axis=0)

    # Lead: 从第二行开始
    lead = x_rep[1:]
    # Lag: 从第一行开始，去掉最后一行
    lag = x_rep[:-1]

    # 拼接成二维路径
    ll_path = np.hstack((lead, lag))
    return ll_path
from sklearn.base import BaseEstimator, TransformerMixin
class AddTime(BaseEstimator, TransformerMixin):
    """Add time component to a single path of shape [L, C]."""
    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        """
        data: numpy.ndarray or torch.Tensor of shape [L, C]
        returns: numpy.ndarray of shape [L, C+1]
        """
        # 转成 torch.Tensor
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        L, C = data.shape

        # 时间通道 [0, 1]，长度为 L
        time_scaled = torch.linspace(0, 1, L, device=data.device).view(L, 1)

        # 拼接在第一列
        out = torch.cat((time_scaled, data), dim=1)

        return out.cpu().numpy()
class AppendZero(BaseEstimator, TransformerMixin):
    """Append a zero starting vector to every path.
    Supports both (L,C) and (B,L,C) inputs.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 如果是 numpy，转成 torch
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # 如果是 (L,C)，扩展 batch 维度
        added_batch = False
        if X.dim() == 2:  # (L,C)
            X = X.unsqueeze(0)  # -> (1,L,C)
            added_batch = True

        B, L, C = X.shape
        zero_vec = torch.zeros(size=(B, 1, C), device=X.device, dtype=X.dtype)
        X_out = torch.cat((zero_vec, X), dim=1)

        # 如果原来是 (L,C)，去掉 batch 维度
        if added_batch:
            X_out = X_out.squeeze(0)  # -> (L+1,C)

        return X_out.cpu().numpy()    

class PenOff(BaseEstimator, TransformerMixin):
    """Adds a 'penoff' dimension to each path. 
    Supports both (B,L,C) and (L,C) inputs.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # 如果输入是二维的 (L,C)，扩展 batch 维度
        added_batch = False
        if X.dim() == 2:  # (L,C)
            X = X.unsqueeze(0)  # -> (1,L,C)
            added_batch = True

        B, L, C = X.shape

        # Add in a dimension of ones# 最前面添加pen状态的一列
        X_pendim = torch.cat((torch.ones(B, L, 1, device=X.device, dtype=X.dtype), X), dim=2)

        # Add pen down to 0
        pen_down = X_pendim[:, [-1], :].clone()
        pen_down[:, :, 0] = 0
        X_pendown = torch.cat((X_pendim, pen_down), dim=1)

        # Add home
        home = torch.zeros(B, 1, C + 1, device=X.device, dtype=X.dtype)
        X_penoff = torch.cat((X_pendown, home), dim=1)

        # 如果原来是 (L,C)，去掉 batch 维度
        if added_batch:
            X_penoff = X_penoff.squeeze(0)  # -> (L+2, C+1)

        return X_penoff.cpu().numpy()
class _Pair:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __iter__(self):
        return iter((self.start, self.end))

class Dyadic:
    """生成 dyadic 窗口划分"""
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, length):
        return self.call(float(length))

    def call(self, length, _offset=0.0, _depth=0, _out=None):
        if _out is None:
            _out = [[] for _ in range(self.depth + 1)]
        _out[_depth].append(_Pair(int(_offset), int(_offset + length)))

        if _depth < self.depth:
            half_length = length / 2
            self.call(half_length, _offset, _depth + 1, _out)
            self.call(half_length, _offset + half_length, _depth + 1, _out)

        return _out


def dyadic_windows_segments(X: torch.Tensor, depth: int):
    """
    输入: 
        X (torch.Tensor): (L,C) 序列
        depth (int): dyadic 划分深度
    输出:
        dict[int, list[torch.Tensor]]: {depth: [片段1, 片段2, ...]}
    """
    assert X.dim() == 2, "输入必须是 (L,C)"
    L, C = X.shape

    dy = Dyadic(depth)
    windows = dy(L)

    result = {}
    for d, wlist in enumerate(windows):
        result[d] = []
        for (s, e) in wlist:
            result[d].append(X[s:e, :])  # (e-s, C)

    return result
def combine_features(lnps,dyn):
    # 对齐时间长度后拼接
    min_len = min(len(lnps), len(dyn))
    return np.concatenate([lnps[:min_len], dyn[:min_len]], axis=-1)
def path_length(path):
    # path: shape (T, d)
    diffs = np.diff(path, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    return np.sum(seg_lengths)
class ShiftToZero:
    """Performs a translation so paths begin at zero (NumPy version).
    Supports both (L, C) and (B, L, C) input.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(X)}")

        if X.ndim == 2:
            # (L, C)
            return X - X[0:1, :]
        elif X.ndim == 3:
            # (B, L, C)
            return X - X[:, [0], :]
        else:
            raise ValueError(f"Unsupported input shape: {X.shape}")

    def fit_transform(self, X, y=None):
        """Combine fit and transform."""
        self.fit(X, y)
        return self.transform(X)
class SubtractBasePoint(BaseEstimator, TransformerMixin):
    """Subtract the first element of each path (base point) from every element.
    Supports both (L,C) and (B,L,C) inputs.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # If the input is a NumPy array, convert it to PyTorch tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # If the input has shape (L, C), add a batch dimension
        added_batch = False
        if X.dim() == 2:  # (L, C)
            X = X.unsqueeze(0)  # -> (1, L, C)
            added_batch = True

        B, L, C = X.shape
        
        # Get the first element (X0) for subtraction
        X0 = X[:, 0, :].unsqueeze(1)  # Shape (B, 1, C)

        # Subtract the base point X0 from every other element
        X_out = X - X0  # Broadcasting applies the subtraction for all elements

        # If the original input was (L, C), remove the batch dimension
        if added_batch:
            X_out = X_out.squeeze(0)  # -> (L, C)

        return X_out.cpu().numpy()  # Convert the tensor back to NumPy array


import torch
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
def sigdims(d, m):
    """返回每一阶的维度长度列表"""
    return [d**k for k in range(1, m+1)]

def sigfeatExt(pathList, feats, dim=2, 
               transform=False, finger_scene=False,
               window_size=10, stride=1, signature_depth=2,
               use_leadlag=False, use_logsig=False, 
                 use_dyadic=False, dyadic_depth=5):
    """
    输入：
        pathList: list[np.ndarray] or list[torch.Tensor] (L,C)
        feats: list 收集特征
    """
    for path in pathList:
        if isinstance(path, torch.Tensor):
            path = path.cpu().numpy()

        p = path[:, dim]  # 额外维度，例如压力

        # 1) 基础几何预处理
        path[:, 0] = bf(path[:, 0])  # X 去噪/滤波
        path[:, 1] = bf(path[:, 1])  # Y 去噪/滤波

        # 去尺度（按总路径长度归一）
        total_len = np.sum(np.sqrt(np.sum(np.diff(path[:, 0:2], axis=0)**2, axis=1))) + 1e-6
        path[:, 0:2] /= total_len
        # dynamic_feat = compute_features23(path)
        dx = diff(path[:, 0]); dy = diff(path[:, 1])
        v = numpy.sqrt(dx**2+dy**2)
        theta = numpy.arctan2(dy, dx)
        cos = numpy.cos(theta)
        sin = numpy.sin(theta)
        dv = diff(v)
        dtheta = numpy.abs(diffTheta(theta))
        logCurRadius = numpy.log((v+0.05) / (dtheta+0.05))
        dv2 = numpy.abs(v*dtheta)
        totalAccel = numpy.sqrt(dv**2 + dv2**2)
        dynamic_feat = np.column_stack((
            dx, dy, v, cos, sin,
            theta, logCurRadius, totalAccel,
            dv, dv2, dtheta, p
        )).astype(np.float32)

        # 3) 归一化
        if finger_scene:
            ''' For finger scenario '''
            dynamic_feat[:,:-1] = (dynamic_feat[:,:-1] - numpy.mean(dynamic_feat[:,:-1], axis=0)) / numpy.std(dynamic_feat[:,:-1], axis=0)
        else:
            ''' For stylus scenario '''
            dynamic_feat = (dynamic_feat - numpy.mean(dynamic_feat, axis=0)) / numpy.std(dynamic_feat, axis=0)
        #print('feat',feat.shape)就是不同的时间长度
        # mu = np.mean(dynamic_feat, axis=0, keepdims=True)
        # sigma = np.std(dynamic_feat, axis=0, keepdims=True) + 1e-6
        # dynamic_feat = (dynamic_feat - mu) / sigma

        # --- 🔹 在这里用你写的预处理模块 --- SubtractBasePoint
        transformer1 = AppendZero()
        dynamic_feat = transformer1.fit_transform(dynamic_feat)  # -> (L+1,C)

        transformer2 = AddTime()
        dynamic_feat = transformer2.fit_transform(dynamic_feat)  # -> (L+1,C+1)

        # 4) 选窗口方式
        signatures = []
        if use_dyadic:  # dyadic 窗口划分
            dy_segments = dyadic_windows_segments(torch.from_numpy(dynamic_feat), depth=dyadic_depth)
            for d, segs in dy_segments.items():
                for seg in segs:
                    if use_logsig:
                        sig = iisignature.logsig(seg.numpy(), iisignature.prepare(seg.shape[1], signature_depth))
                    else:
                        sig = iisignature.sig(seg.numpy(), signature_depth)
                    signatures.append(sig)
        else:  # 滑动窗口
            num_samples = dynamic_feat.shape[0]
            d_dyn = dynamic_feat.shape[1]
            prep_dyn = iisignature.prepare(d_dyn, signature_depth)

            for start in range(0, num_samples - window_size + 1, stride):
                window_dyn = dynamic_feat[start:start + window_size, :]
                if window_dyn.shape[0] == 0 or window_dyn.ndim < 2:
                    continue
                if use_logsig:
                    sig_full = iisignature.logsig(window_dyn, prep_dyn)
                else:
                    sig_full = iisignature.sig(window_dyn, signature_depth)
                # mean_dyn = np.mean(window_dyn, axis=0)
                # feat_combined = np.concatenate([mean_dyn, sig_full], axis=0)

                signatures.append(sig_full)

        signatures = np.array(signatures, dtype=np.float64)
        feats.append(signatures)

    return feats

from scipy.signal import savgol_filter
from scipy.signal import medfilt

class MedianSavGolFilter:
    def __init__(self, median_kernel=3, sg_window=9, sg_poly=3):
        self.median_kernel = median_kernel
        self.sg_window = sg_window
        self.sg_poly = sg_poly
    def __call__(self, data):
        data = medfilt(data, self.median_kernel)
        return savgol_filter(data, self.sg_window, self.sg_poly)

# 用法
mf = MedianSavGolFilter(median_kernel=3, sg_window=9, sg_poly=3)

# def featExt2(pathList, feats, dim=2, transform=False, finger_scene=False,window_size=10,stride = 1,signature_depth=2, use_leadlag=False, use_logsig=True, pca_model=None):
#     for path in pathList:
#         p = path[:, dim]  # 按你原逻辑的附加维度（例如压力）

#         # 1) 基础几何预处理
#         path[:, 0] = bf(path[:, 0])  # X 去噪/滤波
#         path[:, 1] = bf(path[:, 1])  # Y 去噪/滤波
        
#         # 去平移
#         path[:, 0:2] -= path[0, 0:2]
#         # 去尺度（按总路径长度归一）
#         total_len = np.sum(np.sqrt(np.sum(np.diff(path[:, 0:2], axis=0)**2, axis=1))) + 1e-6
#         path[:, 0:2] /= total_len

#         # 2) 计算动态特征
#         dx = diff(path[:, 0]); dy = diff(path[:, 1])
#         v = numpy.sqrt(dx**2+dy**2)
#         theta = numpy.arctan2(dy, dx)
#         cos = numpy.cos(theta)
#         sin = numpy.sin(theta)
#         dv = diff(v)
#         dtheta = numpy.abs(diffTheta(theta))
#         logCurRadius = numpy.log((v+0.05) / (dtheta+0.05))
#         dv2 = numpy.abs(v*dtheta)
#         totalAccel = numpy.sqrt(dv**2 + dv2**2)


#         # 3) 拼动态特征矩阵
#         dynamic_feat = np.column_stack((
#             dx, dy, v, cos, sin,
#             theta, logCurRadius, totalAccel,
#             dv, dv2, dtheta, p
#         )).astype(np.float32)

#         # 4) 标准化（防止信息泄漏，这里的均值/方差最好提前在训练集拟合好）
#         mean_ = np.mean(dynamic_feat, axis=0)
#         std_ = np.std(dynamic_feat, axis=0) + 1e-6
#         if finger_scene:
#             dynamic_feat[:, :-1] = (dynamic_feat[:, :-1] - numpy.mean(dynamic_feat[:, :-1], axis=0)) / numpy.std(dynamic_feat[:, :-1], axis=0)
#         else:
#             dynamic_feat = (dynamic_feat - numpy.mean(dynamic_feat, axis=0)) / numpy.std(dynamic_feat, axis=0)

#         xy_feat = path[:, 0:2].astype(np.float32)

#         # 5) 可选：Lead–Lag 扩展
#         if use_leadlag:
#             xy_feat = np.hstack((xy_feat, xy_feat))  # 简单 lead-lag，可替换成更严谨的构造

#         # 6) 按窗口提取特征
#         num_samples = dynamic_feat.shape[0]
#         signatures = []
#         sigspec = None
#         if use_logsig:
#             # 假设所有窗口的列数一致
#             sigspec = iisignature.prepare(xy_feat.shape[1], signature_depth)

#         for start in range(0, num_samples - window_size + 1, stride):
#             window_xy = xy_feat[start:start + window_size]

#             if window_xy.shape[0] == 0 or window_xy.ndim < 2:
#                 print(f"Skipping invalid window with shape {window_xy.shape}")

#                 continue
#             if use_logsig:
#                 sig = iisignature.logsig(window_xy, sigspec)
#             else:
#                 sig = iisignature.sig(window_xy, signature_depth)

#             # 签名或 log-signature


#             # 其他特征均值
#             rest_feat_mean = np.mean(dynamic_feat[start:start + window_size, :], axis=0)
#             rest_sig_window = iisignature.sig(dynamic_feat[start:start + window_size, :], signature_depth) 

#             sig_full = np.concatenate((sig, rest_feat_mean,rest_sig_window), axis=0)
#             signatures.append(sig_full)

#         signatures = np.array(signatures, dtype=np.float64)

#         # 7) 可选：PCA/降冗（防止全局阈值性能下降）
#         if pca_model is not None:
#             signatures = pca_model.transform(signatures)

#         feats.append(signatures)
#     return feats
import numpy as np
from sklearn.decomposition import PCA

def featExt2(pathList, feats, dim=2, transform=False, finger_scene=False,window_size=10,stride = 1,signature_depth=2, use_leadlag=False, use_logsig=True, pca_model=False):
    for path in pathList:
        p = path[:, dim]  # 按你原逻辑的附加维度（例如压力）

        # 1) 基础几何预处理
        path[:, 0] = bf(path[:, 0])  # X 去噪/滤波
        path[:, 1] = bf(path[:, 1])  # Y 去噪/滤波
        
        # # 去平移
        path[:, 0:2] -= path[0, 0:2]
        # 去尺度（按总路径长度归一）
        total_len = np.sum(np.sqrt(np.sum(np.diff(path[:, 0:2], axis=0)**2, axis=1))) + 1e-6
        path[:, 0:2] /= total_len

        # mu = np.mean(path[:, 2], axis=0, keepdims=True)
        # sigma = np.std(path[:, 2], axis=0, keepdims=True) + 1e-6
        # path[:, 2] = (path[:, 2] - mu) / sigma    
        #     
        dynamic_feat = compute_features23(path)
        # 7) 可选：PCA/降冗（防止全局阈值性能下降）
        if pca_model :
            pca_model = PCA(n_components=10)
            dynamic_feat = pca_model.fit_transform(dynamic_feat)
            #    例如使用 z-score
        
        mu = np.mean(dynamic_feat, axis=0, keepdims=True)
        sigma = np.std(dynamic_feat, axis=0, keepdims=True) + 1e-6
        features_norm = (dynamic_feat - mu) / sigma
        # 这个就不一定要用
        if finger_scene:
            dynamic_feat[:, :-1] = (dynamic_feat[:, :-1] - numpy.mean(dynamic_feat[:, :-1], axis=0)) / numpy.std(dynamic_feat[:, :-1], axis=0)

        else:
            dynamic_feat = (dynamic_feat - numpy.mean(dynamic_feat, axis=0)) / numpy.std(dynamic_feat, axis=0)
        xy_feat = path[:, 0:2].astype(np.float32)

        # 5) 可选：Lead–Lag 扩展
        if use_leadlag:
            xy_feat = np.hstack((xy_feat, xy_feat))  # 简单 lead-lag，可替换成更严谨的构造

        # 6) 按窗口提取特征
        num_samples = dynamic_feat.shape[0]
        signatures = []
        sigspec = None
        if use_logsig:
            # 假设所有窗口的列数一致
            sigspec = iisignature.prepare(xy_feat.shape[1], signature_depth)

        for start in range(0, num_samples - window_size + 1, stride):
            window_xy = xy_feat[start:start + window_size]

            if window_xy.shape[0] == 0 or window_xy.ndim < 2:
                print(f"Skipping invalid window with shape {window_xy.shape}")

                continue
            if use_logsig:
                sig = iisignature.logsig(window_xy, sigspec)
            else:
                sig = iisignature.sig(window_xy, signature_depth)

            # 签名或 log-signature


            # 其他特征均值
            rest_feat_mean = np.mean(dynamic_feat[start:start + window_size, :], axis=0)
            rest_sig_window = iisignature.sig(dynamic_feat[start:start + window_size, :], signature_depth) 

            sig_full = np.concatenate((sig, rest_feat_mean,rest_sig_window), axis=0)
            signatures.append(sig_full)

        signatures = np.array(signatures, dtype=np.float64)



        feats.append(signatures)

    return feats
def compute_features23(path):
    """
    输入: path (N x 3) 矩阵, 列分别为 [x, y, z] (坐标和压力)
    输出: features (N x 23) 特征矩阵
    """

    # --- 基础输入 ---
    x = path[:, 0]
    y = path[:, 1]
    z = path[:, 2] if path.shape[1] > 2 else np.zeros_like(x)

    # 一阶差分
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dz = np.diff(z, prepend=z[0])

    # 二阶差分
    ddx = np.diff(dx, prepend=dx[0])
    ddy = np.diff(dy, prepend=dy[0])

    # 路径速度
    v = np.sqrt(dx**2 + dy**2) + 1e-6

    # 路径切向角
    theta = np.arctan2(dy, dx)

    # 正余弦
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # 角度变化
    dtheta = np.diff(theta, prepend=theta[0])
    dtheta = np.unwrap(dtheta)   # 避免角度跳变

    # 曲率半径对数
    rho = np.log((v + 0.05) / (np.abs(dtheta) + 0.05))

    # 加速度
    dv = np.diff(v, prepend=v[0])
    dv2 = np.abs(v * dtheta)
    a = np.sqrt(dv**2 + dv2**2)

    # ========================
    # 1-7 原始特征
    # ========================
    f1 = x
    f2 = y
    f3 = z
    f4 = theta
    f5 = v
    f6 = rho
    f7 = a

    # ========================
    # 8-14 一阶导数
    # ========================
    f8 = dx
    f9 = dy
    f10 = dz
    f11 = np.gradient(theta)
    f12 = np.gradient(v)
    f13 = np.gradient(rho)
    f14 = np.gradient(a)

    # ========================
    # 15-16 二阶导数
    # ========================
    f15 = ddx
    f16 = ddy

    # ========================
    # 17 局部速度比
    # ========================
    win5 = 5
    v_min = np.array([np.min(v[max(0, i-win5):i+1]) for i in range(len(v))])
    v_max = np.array([np.max(v[max(0, i-win5):i+1]) for i in range(len(v))]) + 1e-6
    f17 = v_min / v_max

    # ========================
    # 18-19 连续角度和差分
    # ========================
    f18 = theta[1:] - theta[:-1]
    f18 = np.concatenate([[f18[0]], f18])   # 补齐长度
    f19 = np.gradient(f18)

    # ========================
    # 20-21 sin/cos
    # ========================
    f20 = sin_t
    f21 = cos_t

    # ========================
    # 22-23 笔画长宽比
    # ========================
    def stroke_ratio(v, win):
        ratios = []
        for i in range(len(x)):
            xw = x[max(0, i-win):i+1]
            yw = y[max(0, i-win):i+1]
            length = np.sum(np.sqrt(np.diff(xw)**2 + np.diff(yw)**2)) + 1e-6
            width = (np.max(xw) - np.min(xw)) + (np.max(yw) - np.min(yw)) + 1e-6
            ratios.append(length / width)
        return np.array(ratios)

    f22 = stroke_ratio(v, 5)
    f23 = stroke_ratio(v, 7)

    # ========================
    # 拼接 23 维特征
    # ========================
    features = np.column_stack([f1,f2,
        f3,f4,f5,f6,f7,f8,
        f9,f10,f11,f12,
        f13,f14,f15,f16,
        f17,f18,f19,
        f20,f21,f22,f23,
    ]).astype(np.float32)

    return features

