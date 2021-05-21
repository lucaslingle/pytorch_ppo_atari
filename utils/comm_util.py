from mpi4py import MPI
import torch as tc


def get_comm():
    comm = MPI.COMM_WORLD
    return comm

def sync_params(agent, comm):
    for p in agent.parameters():
        p_data = p.data.numpy()
        comm.Bcast(p_data, root=0)
        p.copy_(tc.FloatTensor(p_data))

def mpi_mean()