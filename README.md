# Online Signature Verication Using Augmented Path Signature and T-Mamba
This repository contains the official implementation for online signature verification using  Augmented Path Signatures for robust feature extraction and T-Mamba for efficient sequence modeling.
## Key Features
Augmented Path Signatures: Capture fine-grained geometric and analytical properties of signature.

T-Mamba Architecture: To optimize sequence modeling, we introduce T-Mamba, which replaces traditional RNNs/Transformers with a hybrid architecture for superior long-range dependency modeling with linear-time complexity $O(L)$.

High Performance: Optimized for both stylus-based and finger-based online signature datasets including MCYT-100, SVC-2004 Task2, DeepSign DB this three datasets.

## Quick Start
'''
# 1. Training 
python -m DB_train --seed 111 
python -m DB_train --seed 222
python -m DB_train --seed 333
python -m DB_train --seed 444
python -m DB_train --seed 555
# 2. Evaluating 
python -m DB_eva_stylus --seed 111  
python -m DB_eva_stylus --seed 222
python -m DB_eva_stylus --seed 333
python -m DB_eva_stylus --seed 444
python -m DB_eva_stylus --seed 555
# 3. Final verification and EER calculation
python -m DB_verify_stylus 
'''
