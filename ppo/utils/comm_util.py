from mpi4py import MPI
import torch as tc
import numpy as np


def get_comm():
    comm = MPI.COMM_WORLD
    return comm


@tc.no_grad()
def sync_state(agent, comm, root):
    model_state_dict = comm.bcast(agent.model.state_dict(), root=root)
    optimizer_state_dict = comm.bcast(agent.optimizer.state_dict(), root=root)
    scheduler_state_dict = comm.bcast(agent.scheduler.state_dict(), root=root)

    agent.model.load_state_dict(model_state_dict)
    agent.optimizer.load_state_dict(optimizer_state_dict)
    agent.scheduler.load_state_dict(scheduler_state_dict)


@tc.no_grad()
def sync_grads(model, comm):
    for p in model.parameters():
        p_grad_local = p.grad.numpy()
        p_grad_global = np.zeros_like(p_grad_local)
        comm.Allreduce(sendbuf=p_grad_local, recvbuf=p_grad_global, op=MPI.SUM)
        p.grad.copy_(tc.FloatTensor(p_grad_global) / comm.Get_size())
