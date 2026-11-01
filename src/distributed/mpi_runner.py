from mpi4py import MPI


class MPIRunner:

    def __init__(self):

        self.comm = MPI.COMM_WORLD

        self.rank = (
            self.comm.Get_rank()
        )

        self.size = (
            self.comm.Get_size()
        )

    def distribute(
        self,
        items,
    ):

        return list(items)[
            self.rank :: self.size
        ]

    def gather(
        self,
        value,
    ):

        return self.comm.gather(
            value,
            root=0,
        )

    def barrier(self):

        self.comm.Barrier()