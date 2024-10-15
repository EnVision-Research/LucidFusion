import os, sys
import shutil
import datetime
import torch
import torch.nn.functional as torch_F
import socket
import contextlib
import torch.distributed as dist
# from collections import defaultdict, deque

# move tensors to device in-place
def move_to_device(X, device):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = move_to_device(v, device)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = move_to_device(e, device)
    elif isinstance(X, tuple) and hasattr(X, "_fields"): # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd, device)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.to(device=device, non_blocking=True)
    return X

# this recursion seems to only work for the outer loop when dict_type is not dict
def to_dict(D, dict_type=dict):
    D = dict_type(D)
    for k, v in D.items():
        if isinstance(v, dict):
            D[k] = to_dict(v, dict_type)
    return D

def get_child_state_dict(state_dict, key):
    out_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            param_name = k[7:]
        else:
            param_name = k
        if param_name.startswith("{}.".format(key)):
            out_dict[".".join(param_name.split(".")[1:])] = v
    return out_dict

def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open

def get_layer_dims(layers):
    # return a list of tuples (k_in, k_out)
    return list(zip(layers[:-1], layers[1:]))

@contextlib.contextmanager
def suppress(stdout=False, stderr=False):
    with open(os.devnull, "w") as devnull:
        if stdout: old_stdout, sys.stdout = sys.stdout, devnull
        if stderr: old_stderr, sys.stderr = sys.stderr, devnull
        try: yield
        finally:
            if stdout: sys.stdout = old_stdout
            if stderr: sys.stderr = old_stderr

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_grad2(d_outs, x_in):
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    reg = 0
    for d_out in d_outs:
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg += grad_dout2.view(batch_size, -1).sum(1)
    return reg / len(d_outs)

def cleanup():
    dist.destroy_process_group()

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def setup(rank, world_size, port_no):
    full_address = 'tcp://127.0.0.1:' + str(port_no)
    # dist.init_process_group("nccl", init_method=full_address, rank=rank, world_size=world_size)
    dist.init_process_group("nccl", init_method=full_address, rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600))

def print_grad(grad, prefix=''):
    print("{} --- Grad Abs Mean, Grad Max, Grad Min: {:.5f} | {:.5f} | {:.5f}".format(prefix, grad.abs().mean().item(), grad.max().item(), grad.min().item()))
        
class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)

