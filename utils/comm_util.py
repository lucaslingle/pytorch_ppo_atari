from mpi4py import MPI
import torch as tc
import numpy as np


def get_comm():
    comm = MPI.COMM_WORLD
    return comm

def sync_params(agent, comm):
    for p in agent.parameters():
        p_data = p.data.numpy()
        comm.Bcast(p_data, root=0)
        p.copy_(tc.FloatTensor(p_data))

def sync_grads(agent, comm):
    for p in agent.parameters():
        p_grad_local = p.grad.numpy()
        p_grad_global = np.zeros_like(p_grad_local)
        comm.Allreduce(sendbuf=p_grad_local, recvbuf=p_grad_global, op=MPI.SUM)
        p.grad.copy_(tc.FloatTensor(p_grad_global) / comm.Get_size())
