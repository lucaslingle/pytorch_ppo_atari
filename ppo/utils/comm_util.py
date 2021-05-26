from mpi4py import MPI
import torch as tc
import numpy as np


def get_comm():
    comm = MPI.COMM_WORLD
    return comm


@tc.no_grad()
def sync_params(model, comm, root):
    for p in model.parameters():
        p_data = p.data.numpy()
        comm.Bcast(p_data, root=root)
        p.copy_(tc.FloatTensor(p_data))


@tc.no_grad()
def sync_grads(model, comm):
    for p in model.parameters():
        p_grad_local = p.grad.numpy()
        p_grad_global = np.zeros_like(p_grad_local)
        comm.Allreduce(sendbuf=p_grad_local, recvbuf=p_grad_global, op=MPI.SUM)
        p.grad.copy_(tc.FloatTensor(p_grad_global) / comm.Get_size())
