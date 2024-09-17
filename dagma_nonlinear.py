from locally_connected import LocallyConnected
import torch
import torch.nn as nn
import numpy as np
from  torch import optim
import copy
from tqdm.auto import tqdm
from math import log, pi

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def adjacency_matrix_to_dict(W, dont_include_self_loops=True):
    '''
    Returns a dictionary of the form {'W[i][j]': W[i,j]} for all i,j
    '''
    W_dict = {}
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if dont_include_self_loops:
                if i != j:
                    W_dict[f'W[{i}][{j}]'] = W[i,j]
    return W_dict

class Spindly(nn.Module):
    def __init__(self, in_features):
        super(Spindly, self).__init__()
        self.weight = nn.Parameter(torch.zeros(in_features))
        
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0)
        
    def forward(self, input):
        return input * self.weight


class SEM_Dataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx]
    

def stable_softplus(input):
   return nn.functional.softplus(input) + 1e-8


def get_head_activation(act_str):
    if act_str == 'softplus':
        return stable_softplus
    elif act_str == 'exp':
        return torch.exp
    else:
        raise ValueError('Invalid activation string', act_str)
    

def get_hidden_activation(act_str):
    if act_str == 'GELU':
        return torch.nn.GELU()
    elif act_str == 'ReLU':
        return torch.nn.ReLU()
    elif act_str == 'sigmoid':
        return torch.nn.Sigmoid()
    else:
        raise ValueError('Invalid activation string', act_str)


class DagmaNN(nn.Module):
    
    def __init__(self, dims, bias=True, LSN=False, natural_param=False, hidden_act_str='GELU', variance_act_str='softplus'):
        super(DagmaNN, self).__init__()
        assert len(dims) >= 2
        self.tracker = []
        self.LSN = LSN
        self.hidden_activation = get_hidden_activation(hidden_act_str)
        self.natural_param = natural_param
        if self.LSN:
            assert dims[-1] == 2
            self.head_activation_variance = get_head_activation(variance_act_str)
        else:
            assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)

        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        
    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = self.hidden_activation(x)
            x = fc(x)

        if self.LSN:
            f0 = x[:,:,0]
            f1 = self.head_activation_variance(x[:,:,1]) # mu, sigma^2 [i, j] i=1,..,n j=1,...,d
            if self.natural_param:
                eta_1 = f0
                eta_2 = -0.5 * f1
                return eta_1, eta_2
            else:
                return f0, f1
        else:
            return x.squeeze(dim=2) # mu [i, j] i=1,..,n j=1,...,d

    def h_func(self, s=1.0):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
        h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        return torch.sum(torch.abs(self.fc1.weight))
    
    def fc1_l1_group_lasso(self):
        """Take group lasso penalty where the groups are the columns of A_j^{1}"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  
        A = torch.sum(fc1_weight ** 2, dim=1).t() 
        W = torch.sqrt(A)

        return torch.sum(A)

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  
        A = torch.sum(fc1_weight ** 2, dim=1).t() 
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class DagmaNN_no_self_loops(nn.Module):
    
    def __init__(self, dims, bias=True, LSN=False, natural_param=False, hidden_act_str='GELU',
                 variance_act_str='softplus', spindly_layer=False, edge_clamp_range=None):
        super(DagmaNN_no_self_loops, self).__init__()
        assert len(dims) >= 2
        self.tracker = []
        self.spindly_layer = spindly_layer
        self.LSN = LSN
        self.hidden_activation = get_hidden_activation(hidden_act_str)
        self.natural_param = natural_param
        if self.LSN:
            assert dims[-1] == 2
            self.head_activation_variance = get_head_activation(variance_act_str)
        else:
            assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)
        self.edge_clamp_range_init = edge_clamp_range
        self.edge_clamp_range = self.edge_clamp_range_init
        self.to_keep = torch.ones((self.d, self.d)) - torch.eye(self.d) # remeber edges that are clamped by setting them to zero

        # it is required for barrier methods to start at the interior
        # of the feasibility region, thus W has to be a DAG
        # (trivially true if all weights are zero). 
        if self.spindly_layer:
            # spindly layer
            self.spindly_list = []
            for i in range(self.d):
                self.spindly_list.append(Spindly(in_features=self.d-1))
                self.spindly_list = nn.ModuleList(self.spindly_list)
            # fc1: one additional local linear layer needed after the spindly layer. 
            self.fc1 = nn.ModuleList([LocallyConnected(self.d, self.d - 1, dims[1], bias=bias)])
        else:
            # special fc1 layer that does not induce self-loops
            layers = []
            for i in range(self.d):
                fc = torch.nn.Linear(in_features=self.d - 1, out_features=dims[1], bias=True)
                nn.init.zeros_(fc.weight)
                nn.init.zeros_(fc.bias)
                layers.append(fc)
            self.special_fc1 = nn.ModuleList(layers)
        
        # fc2: local linear layers
        self.batch_norm_layers = [] # for every hidden layer we have d local batch layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
            self.batch_norm_layers.append([nn.BatchNorm1d(self.dims[l + 1]) for _ in range(self.d)]) 
        self.fc2 = nn.ModuleList(layers)
        
    def forward(self, x):  # [n, d] -> [n, d]
        if self.spindly_layer:
            # spindly layer
            out = [None] * self.d
            input_idxes = list(range(self.d))
            for i in range(self.d):
                input_idx = input_idxes.copy()
                input_idx.remove(i)
                out[i] = self.spindly_list[i](x[:, input_idx])
            x = torch.cat(out, 1)
            # fc1
            x = x.view(-1, self.d, self.d - 1)
            x = self.fc1[0](x)
        else:
            # special fc1 layer
            out = [None] * self.d
            input_idxes = list(range(self.d))
            for i in range(self.d):
                input_idx = input_idxes.copy()
                input_idx.remove(i)
                out[i] = self.special_fc1[i](x[:, input_idx])
            x = torch.cat(out, 1)
            x = x.view(-1, self.d, self.dims[1])

        # fc2
        for l, fc in enumerate(self.fc2):
            
            ## batch norm
            #outputs = []
            #for d, batch_norm in enumerate(self.batch_norm_layers[l]):
            #    outputs.append(batch_norm(x[:,d,:]))
            #x = torch.stack(outputs, dim=1)

            x = self.hidden_activation(x)
            x = fc(x)

        if self.LSN:
            f0 = x[:,:,0]
            f1 = self.head_activation_variance(x[:,:,1]) # mu, sigma^2 [i, j] i=1,..,n j=1,...,d
            if self.natural_param:
                eta_1 = f0
                eta_2 = -0.5 * f1
                return eta_1, eta_2
            else:
                return f0, f1
        else:
            return x.squeeze(dim=2) # mu [i, j] i=1,..,n j=1,...,d
    
    def h_func(self, s=1.0):
        """Constrain weighted adjacency matrics to be a DAG"""
        if self.spindly_layer:
            A = torch.zeros([self.d, self.d])
            input_idxes = list(range(self.d))
            for i in range(self.d):
                input_idx = input_idxes.copy()
                input_idx.remove(i)
                A[input_idx, i] = self.spindly_list[i].weight ** 2
        else: 
            A = torch.zeros([self.d, self.d])
            input_idxes = list(range(self.d))
            for i in range(self.d):
                input_idx = input_idxes.copy()
                input_idx.remove(i)
                A[input_idx, i] = torch.sum(self.special_fc1[i].weight ** 2, dim=0)
        
        # clamp edges
        with torch.no_grad():
            to_keep_ = (torch.sqrt(A) >= self.edge_clamp_range).type(torch.Tensor)
            self.to_keep *= to_keep_
            A *= self.to_keep
        h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weights"""
        penalty = 0
        if self.spindly_layer:
            # spindly layer
            for spindly in self.spindly_list:
                penalty += torch.sum(torch.abs(spindly.weight))
            # fc1
            penalty += torch.sum(torch.abs(self.fc1[0].weight))
        else:
            for layer in self.special_fc1:
                penalty += torch.sum(torch.abs(layer.weight))
        return penalty
    
    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [list of weights] -> [i, j]
        """Get W from neural networks weights"""
        if self.spindly_layer:
            # Get W from Spindly-layer weights
            A = torch.zeros([self.d, self.d])
            input_idxes = list(range(self.d))
            for i in range(self.d):
                input_idx = input_idxes.copy()
                input_idx.remove(i)
                A[input_idx, i] = self.spindly_list[i].weight ** 2
        else:
            # Get W from fc1 weights
            A = torch.zeros([self.d, self.d])
            input_idxes = list(range(self.d))
            for i in range(self.d):
                input_idx = input_idxes.copy()
                input_idx.remove(i)
                A[input_idx, i] = torch.sum(self.special_fc1[i].weight ** 2, dim=0)
        W = torch.sqrt(A)
        to_keep_ = (W >= self.edge_clamp_range).type(torch.Tensor)
        self.to_keep *= to_keep_
        W *= self.to_keep
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def log_mse_loss(output, target):
    n, d = target.shape
    loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
    return loss


def heteroscedastic_mse_loss(input, target, reduction='mean'):
    """Heteroscedastic negative log likelihood Normal.
    Parameters
    ----------
    input : torch.Tensor (n, 2)
        two natural parameters per data point
    target : torch.Tensor (n, 1)
        targets
    reduction : str
        either 'mean' or 'sum'
    returns
    -------
    torch.Tensor
        negative log likelihood
    """
    assert input.ndim == target.ndim == 2
    assert input.shape[0] == target.shape[0]
    n, _ = input.shape
    C = - 0.5 * log(2 * pi)
    target = torch.cat([target, target.square()], dim=1)
    inner = torch.einsum('nk,nk->n', target, input)
    log_A = input[:, 0].square() / (4 * input[:, 1]) + 0.5 * torch.log(- 2 * input[:, 1])
    log_lik = n * C + inner.sum() + log_A.sum()
    if reduction == 'mean':
        return - log_lik / n
    elif reduction == 'sum':
        return - log_lik 
    else:
        raise ValueError('Invalid reduction', reduction)
    

def beta_Gaussian_NLLLoss(input, target, var, beta=0.5, reduction='mean', full=True, eps=1e-6):
    """Beta-Gaussian negative log likelihood.
    Parameters
    ----------
    input : torch.Tensor (n, 1)
        mean predictions
    target : torch.Tensor (n, 1)
        mean targets
    var : torch.Tensor (n, 1)
        variance predictions
    beta : float
        between 0 and 1 
    reduction : str
        either 'mean' or 'sum'
    full : bool
        whether to include the constant term in the loss
    eps : float
        lower bound for variance
    returns
    -------
    torch.Tensor
        The loss.
    """
    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")
    
    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var)
    if full:
        loss += 0.5 * log(2 * pi)
    loss = loss * var.detach() ** beta

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()


def generic_loss(model, Xb, beta_NLL, s, group_lasso, lambda1, mu, homo_loss=None):
    """Calculates the loss corresponding the the right model and parametrization.
    Parameters
    ----------
    model : Model
        The model to calculate the loss for.
    Xb : torch.Tensor
        Batch of design matrix
    beta_NLL : float
        The beta parameter for the beta-Gaussian likelihood.
    s : int
        Specific DAGMA penalty.
    group_lasso : bool
        Specifies whether group lasso (True) or the lasso penalty (False) is used.
    lambda1 : float
        The weight for the lasso penalty.
    mu : float
        The weight for the score.
    homo_loss : str, optional
        The type of homoscedastic loss to use. One of 'log_mse' or 'NLL'. Defaults to None.

    Returns
    -------
    torch.Tensor
        The loss.
    """
    assert beta_NLL >= 0
    ## likelihood
    if model.LSN:
        d = Xb.shape[1]
        neg_lik = 0
        if model.natural_param:
            eta_1, eta_2 = model(Xb)
            for j in range(d):
                neg_lik += heteroscedastic_mse_loss(
                    input=torch.stack([eta_1[:,j], eta_2[:,j]], 1),
                    target=Xb[:,j].unsqueeze(-1),
                    reduction='mean'
                )
            neg_lik /= d
        else:
            f0, f1 = model(Xb)
            for j in range(d):
                neg_lik += beta_Gaussian_NLLLoss(
                    input=f0[:,j].unsqueeze(-1),
                    target=Xb[:,j].unsqueeze(-1),
                    var=f1[:,j].unsqueeze(-1),
                    beta=beta_NLL, reduction='mean', eps=1e-6
                )
            neg_lik /= d
    else:
        f0 = model(Xb)
        if homo_loss == 'log_mse':
            neg_lik = log_mse_loss(f0, Xb)
        elif homo_loss == 'NLL':
            neg_lik = -torch.distributions.normal.Normal(loc=f0, scale=1).log_prob(Xb).mean()
        else:
            raise ValueError('Invalid homoscedastic loss', homo_loss)

    ## penalties
    h_val = model.h_func(s)
    if group_lasso:
        l1_penalty = lambda1 * model.fc1_l1_group_lasso()
    else:
        l1_penalty = lambda1 * model.fc1_l1_reg()
    
    obj = mu * (neg_lik + l1_penalty) + h_val
    return obj, neg_lik, h_val, l1_penalty


def minimize(model, X, max_iter, t, lr, lambda1, lambda2, group_lasso, mu, s, beta_NLL, batch_size, 
             homo_loss=None, verbose=False, pbar=None, track_optimization=False, optimizer_type=False):
    vprint = print if verbose else lambda *a, **k: None
    vprint(f'\nMinimize s={s} -- lr={lr}')
    if optimizer_type == 'DAGMA-ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
    elif optimizer_type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=mu*lambda2)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())
    else:
        raise ValueError('Invalid optimizer_type: ', optimizer_type)

    train_data = SEM_Dataset(X)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #if lr_decay is True:
    #    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8) 
    for i in range(max_iter):
        model.train()
        # first iteration no edge clamping
        if t==0:
            model.edge_clamp_range = 0
        else:
            model.edge_clamp_range = model.edge_clamp_range_init
        for Xb in train_dataloader:
            obj_b, _, h_val_b, _ = generic_loss(model, Xb, beta_NLL, s, group_lasso, lambda1, mu, homo_loss)
            if h_val_b.item() < -0.000001:
                vprint(f'Found h negative {h_val_b.item()} at iter {i}')
                return False
            obj_b.backward()
            optimizer.step()
            optimizer.zero_grad()

        # stores performance after performing epoch i 
        if track_optimization:
            model.eval()
            obj, neg_lik, h_val, _ = generic_loss(model, X, beta_NLL, s, group_lasso, lambda1, mu, homo_loss)
            perf_dict = {
                "t": t,
                "epoch": i,
                "obj": obj.item(),
                "neg_lik": neg_lik.item(),
                "h_val": h_val.item(),
            } 
            perf_dict = {**perf_dict, **adjacency_matrix_to_dict(model.fc1_to_adj())}
            model.tracker.append(perf_dict)

    # store final values
    model.eval()
    obj, neg_lik, h_val, l1_penalty = generic_loss(model, X, beta_NLL, s, group_lasso, lambda1, mu, homo_loss)
    obj_new = obj.item()
    model.loss_store = {"obj": obj_new, "neg_lik": neg_lik.item(), "h_val": h_val.item(), "l1_penalty": l1_penalty.item()}

    pbar.update(max_iter)
    return True


def dagma_nonlinear(
        model: nn.Module, X: torch.tensor, lambda1=.02, lambda2=.005, group_lasso=False,
        T=4, mu_init=.1, mu_factor=.1, s=1.0, warm_iter=5e4, max_iter=8e4, lr=.0002,
        beta_NLL=0, homo_loss=None, batch_size=None, lr_outer_iter_decay=False,
        w_threshold=0.3, verbose=False, track_optimization=False, track_models=False, 
        optimizer_type='ADAM'
    ):
    vprint = print if verbose else lambda *a, **k: None
    mu = mu_init
    if type(s) == list:
        if len(s) < T: 
            vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
            s = s + (T - len(s)) * [s[-1]]
    elif type(s) in [int, float]:
        s = T * [s]
    else:
        ValueError("s should be a list, int, or float.") 
    with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
        model_tracker = []
        if track_models:
            model_tracker.append({'t': 'init', 'model': copy.deepcopy(model)}) # save initial model
        for t in range(int(T)):
            vprint(f'\nDagma iter t={t+1} -- mu: {mu}', 30*'-')
            success, s_cur = False, s[t]
            inner_iter = int(max_iter) if t == T - 1 else int(warm_iter)
            model_copy = copy.deepcopy(model)
            while success is False:
                success = minimize(
                    model, X, inner_iter, t, lr, lambda1, lambda2, group_lasso, mu, s_cur, beta_NLL, batch_size,
                    homo_loss=homo_loss, verbose=verbose, pbar=pbar, track_optimization=track_optimization,
                    optimizer_type=optimizer_type
                )
                if success is False:
                    model.load_state_dict(model_copy.state_dict().copy())
                    lr *= 0.5
                    if lr < 1e-10:
                        break # lr is too small
                    s_cur = 1
            
            if lr_outer_iter_decay:
                lr *= 0.9
            mu *= mu_factor
            if track_models:
                model_tracker.append({'t': t, 'model': copy.deepcopy(model)})
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, model_tracker


if __name__ == '__main__':
    from timeit import default_timer as timer
    import utils
    
    torch.set_default_dtype(torch.double)
    utils.set_random_seed(1)
    torch.manual_seed(1)
    
    n, d, s0, graph_type, sem_type = 1000, 20, 20, 'ER', 'mlp'
    B_true = utils.simulate_dag(d, s0, graph_type)
    X = utils.simulate_nonlinear_sem(B_true, n, sem_type)

    model = DagmaNN(dims=[d, 10, 1], bias=True)
    X_torch = torch.from_numpy(X)
    tstart = timer()
    W_est, _ = dagma_nonlinear(model, X_torch, lambda1=0.02, lambda2=0.005)
    tend = timer()
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(f'runtime: {tend-tstart}')
    print(acc)