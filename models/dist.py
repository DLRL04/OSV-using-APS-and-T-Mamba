#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DTW Distance Computation Module

This module computes Dynamic Time Warping (DTW) distances between signature sequences
for signature verification tasks. It handles both skilled forgeries and random forgeries.
"""

import numpy as np
from models.fastdtw import fastdtw as dtw


def dist_seq(FEAT_SEQ, ng, nf, num_g, num_f):
    """
    Compute DTW distances between anchor, positive (genuine), and negative (forged) signatures.
    
    This function calculates pairwise DTW distances for signature verification:
    - Anchor vs Positive: Genuine signatures from the same user
    - Anchor vs Negative: Skilled forgery signatures
    
    Args:
        FEAT_SEQ (list): List of feature sequences for each user.
                        Each sequence contains [anchors, positives, negatives]
        ng (int): Number of anchor (reference) signatures
        nf (int): Number of skilled forgery signatures
        num_g (int): Total number of genuine signatures
        num_f (int): Total number of forgery signatures
    
    Returns:
        tuple: (DIST_P, DIST_N, DIST_TEMP)
            - DIST_P (np.ndarray): Distances between anchors and positive samples
            - DIST_N (np.ndarray): Distances between anchors and negative samples
            - DIST_TEMP (np.ndarray): Pairwise distances among anchors
    """
    DIST_P = []
    DIST_N = []
    DIST_TEMP = []
    
    print("Calculating DTW distance...")
    
    for idx, feat_seq in enumerate(FEAT_SEQ):
        # Extract anchor, positive, and negative samples
        feat_a = feat_seq[0:ng]  # Anchor signatures
        feat_p = feat_seq[(ng + nf):(num_g + nf)]  # Positive (genuine) signatures
        feat_n = feat_seq[(num_g + nf):]  # Negative (forged) signatures
        
        # Initialize distance matrices
        dist_p = np.zeros((num_g - ng, ng))
        dist_n = np.zeros((num_f, ng))
        dist_temp = np.zeros((ng, ng))
        
        # Compute pairwise distances among anchors
        for i in range(ng):
            fp = feat_a[i]
            fps = np.sum(fp, axis=1)
            fp = np.delete(fp, np.where(fps == 0)[0], axis=0)
            
            for j in range(i + 1, ng):
                fa = feat_a[j]
                fas = np.sum(fa, axis=1)
                fa = np.delete(fa, np.where(fas == 0)[0], axis=0)
                
                dist, path = dtw(fp, fa, radius=2, dist=1)
                dist = dist / (fp.shape[0] + fa.shape[0])
                dist_temp[i, j] = dist
        
        # Compute distances between anchors and positive samples
        for i in range(num_g - ng):
            fp = feat_p[i]
            fps = np.sum(fp, axis=1)
            fp = np.delete(fp, np.where(fps == 0)[0], axis=0)
            
            for j in range(ng):
                fa = feat_a[j]
                fas = np.sum(fa, axis=1)
                fa = np.delete(fa, np.where(fas == 0)[0], axis=0)
                
                dist, path = dtw(fp, fa, radius=2, dist=1)
                dist = dist / (fp.shape[0] + fa.shape[0])
                dist_p[i, j] = dist
        
        # Compute distances between anchors and negative samples
        for i in range(num_f):
            fn = feat_n[i]
            fns = np.sum(fn, axis=1)
            fn = np.delete(fn, np.where(fns == 0)[0], axis=0)
            
            for j in range(ng):
                fa = feat_a[j]
                fas = np.sum(fa, axis=1)
                fa = np.delete(fa, np.where(fas == 0)[0], axis=0)
                
                dist, path = dtw(fn, fa, radius=2, dist=1)
                dist = dist / (fn.shape[0] + fa.shape[0])
                dist_n[i, j] = dist
        
        DIST_P.append(dist_p)
        DIST_N.append(dist_n)
        DIST_TEMP.append(dist_temp)
    
    # Concatenate all distance matrices
    DIST_P = np.concatenate(DIST_P, axis=0)
    DIST_N = np.concatenate(DIST_N, axis=0)
    DIST_TEMP = np.concatenate(DIST_TEMP, axis=0)
    
    return DIST_P, DIST_N, DIST_TEMP


def dist_seq_rf(FEAT_SEQ, ng, nf, num_g, num_f):
    """
    Compute DTW distances with random forgeries (cross-user negative samples).
    
    This function extends dist_seq() by using genuine signatures from other users
    as random forgery samples, enhancing model generalization capability.
    
    Random forgeries are created by using genuine signatures from different users,
    simulating attacks where the forger doesn't know the target signature style.
    
    Args:
        FEAT_SEQ (list): List of feature sequences for each user.
                        Each sequence contains [anchors, positives, negatives]
        ng (int): Number of anchor (reference) signatures
        nf (int): Number of skilled forgery signatures
        num_g (int): Total number of genuine signatures
        num_f (int): Total number of forgery signatures
    
    Returns:
        tuple: (DIST_P, DIST_N, DIST_TEMP)
            - DIST_P (np.ndarray): Distances between anchors and positive samples
            - DIST_N (np.ndarray): Distances between anchors and random forgery samples
            - DIST_TEMP (np.ndarray): Pairwise distances among anchors
    """
    DIST_P = []
    DIST_N = []
    DIST_TEMP = []
    FEAT_A = []
    FEAT_P = []
    
    # Extract and store anchor and positive features for all users
    for idx, feat_seq in enumerate(FEAT_SEQ):
        feat_a = feat_seq[0:ng]
        feat_p = feat_seq[(ng + nf):(num_g + nf)]
        FEAT_A.append(feat_a)
        FEAT_P.append(feat_p)
    
    del FEAT_SEQ
    
    print("Calculating DTW distance...")
    
    for idx, feat_a in enumerate(FEAT_A):
        feat_p = FEAT_P[idx]
        
        # Create random forgeries from other users' genuine signatures
        feat_n = []
        for i in range(len(FEAT_A)):
            if i != idx:
                # Use the 3rd genuine signature from other users as random forgery
                feat_n.append(FEAT_P[i][2])
        
        # Initialize distance matrices
        dist_p = np.zeros((feat_p.shape[0], ng))
        dist_n = np.zeros((len(feat_n), ng))
        dist_temp = np.zeros((ng, ng))
        
        # Compute pairwise distances among anchors
        for i in range(ng):
            fp = feat_a[i]
            fps = np.sum(fp, axis=1)
            fp = np.delete(fp, np.where(fps == 0)[0], axis=0)
            
            for j in range(i + 1, ng):
                fa = feat_a[j]
                fas = np.sum(fa, axis=1)
                fa = np.delete(fa, np.where(fas == 0)[0], axis=0)
                
                dist, path = dtw(fp, fa, radius=2, dist=1)
                dist = dist / (fp.shape[0] + fa.shape[0])
                dist_temp[i, j] = dist
        
        # Compute distances between anchors and positive samples
        for i in range(feat_p.shape[0]):
            fp = feat_p[i]
            fps = np.sum(fp, axis=1)
            fp = np.delete(fp, np.where(fps == 0)[0], axis=0)
            
            for j in range(ng):
                fa = feat_a[j]
                fas = np.sum(fa, axis=1)
                fa = np.delete(fa, np.where(fas == 0)[0], axis=0)
                
                dist, path = dtw(fp, fa, radius=2, dist=1)
                dist = dist / (fp.shape[0] + fa.shape[0])
                dist_p[i, j] = dist
        
        # Compute distances between anchors and random forgery samples
        for i in range(len(feat_n)):
            fn = feat_n[i]
            fns = np.sum(fn, axis=1)
            fn = np.delete(fn, np.where(fns == 0)[0], axis=0)
            
            for j in range(ng):
                fa = feat_a[j]
                fas = np.sum(fa, axis=1)
                fa = np.delete(fa, np.where(fas == 0)[0], axis=0)
                
                dist, path = dtw(fn, fa, radius=2, dist=1)
                dist = dist / (fn.shape[0] + fa.shape[0])
                dist_n[i, j] = dist
        
        DIST_P.append(dist_p)
        DIST_N.append(dist_n)
        DIST_TEMP.append(dist_temp)
    
    # Concatenate all distance matrices
    DIST_P = np.concatenate(DIST_P, axis=0)
    DIST_N = np.concatenate(DIST_N, axis=0)
    DIST_TEMP = np.concatenate(DIST_TEMP, axis=0)
    
    return DIST_P, DIST_N, DIST_TEMP


