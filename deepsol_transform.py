import pandas as pd
import numpy as np
import os


# Load in data
data_dir = "sameerkhurana10-DSOL_rv0.2-20562ad/data/"

train_src = np.loadtxt(data_dir + "train_src", dtype=str)
val_src = np.loadtxt(data_dir + "val_src", dtype=str)
test_src = np.loadtxt(data_dir + "test_src", dtype=str)
train_tgt = np.loadtxt(data_dir + "train_tgt").astype(int)
val_tgt = np.loadtxt(data_dir + "val_tgt").astype(int)
test_tgt = np.loadtxt(data_dir + "test_tgt").astype(int)

train = np.vstack([train_src, train_tgt]).T
val = np.vstack([val_src, val_tgt]).T
test = np.vstack([test_src, test_tgt]).T

pd_train = pd.DataFrame(train)
pd_val = pd.DataFrame(val)
pd_test = pd.DataFrame(test)


outdir = '.data'
if not os.path.exists(outdir):
    os.mkdir(outdir)

outdir = '.data/deepsol'
if not os.path.exists(outdir):
    os.mkdir(outdir)

pd_train.to_csv(".data/deepsol/train.tsv", sep='\t', header=None, index=False)
pd_val.to_csv(".data/deepsol/val.tsv", sep='\t', header=None, index=False)
pd_test.to_csv(".data/deepsol/test.tsv", sep='\t', header=None, index=False)

