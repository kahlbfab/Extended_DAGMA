import numpy as np
import torch
import pandas as pd

from dagma_nonlinear import DagmaNN_no_self_loops, dagma_nonlinear

torch.set_default_dtype(torch.double)

# configure model
model = 'DAGMA-LSN-natural' #['DAGMA-adj', 'DAGMA-LSN', 'DAGMA-LSN-natural']
fc1_type = 'no_self_loops'
nbr_hidden_units = 50
hidden_act_str = 'GELU'
variance_act_str = 'softplus'
edge_clamp_range=0.1
spindly_layer=False

# configure training
lambda1 = 0.0025 #0.02
lambda2 = 0
batch_size = 200
lr = 0.005
lr_outer_iter_decay = True
mu_init = 1000
mu_factor = 0.6
verbose = 'N'
warm_iter = 100#1000
max_iter = 1000#5000
T = 30#30
optimizer_type = 'ADAM'
train_val_split = False
homo_loss = False
beta_NLL = 0
track_optimization = False
track_models = False # track models after every iteration

# problem specific inputs
path_X = '1000_10_10_ER_LS-s_1_S_1.0_gauss_X.npy'
path_B_true = '1000_10_10_ER_LS-s_1_S_1.0_gauss_B-true.npy'

X = np.load(path_X, allow_pickle=True)
B = np.load(path_B_true, allow_pickle=True)
d = X.shape[1]
dims = [d, 50]


X_torch = torch.from_numpy(X)

if model == 'DAGMA-LSN-natural':
    dims.append(2) # output dim has to be two: mean and log-variance
    model = DagmaNN_no_self_loops(dims=dims, LSN=True, natural_param=True, hidden_act_str=hidden_act_str, variance_act_str=variance_act_str, spindly_layer=spindly_layer, edge_clamp_range=edge_clamp_range)

W_est, model_tracker = dagma_nonlinear(model, X_torch, lambda1=lambda1, lambda2=lambda2, mu_init=mu_init, mu_factor=mu_factor, w_threshold=0, batch_size=batch_size, homo_loss=homo_loss, lr=lr, lr_outer_iter_decay=lr_outer_iter_decay, optimizer_type=optimizer_type, beta_NLL=beta_NLL, verbose=verbose, warm_iter=warm_iter, max_iter=max_iter, T=T, track_optimization=track_optimization, track_models=track_models)

if track_optimization:
    tracker_df = pd.DataFrame(model.tracker)
    tracker_df.to_csv('1000_10_10_ER_LS-s_1_S_1.0_gauss' + '_tracker.csv', index=False)


print(W_est)
print(B)