class Config:
    def __init__(
        self,
        m,
        l,
        g,
        k1,
        k2,
        k_d,
        k_p,
        k_d_bo,
        k_p_bo,
        pi,
        n_iterations,
        x0,
        n_simulation,
    ):
        self.m = m
        self.l = l
        self.k1 = k1
        self.k2 = k2
        self.k_d = k_d
        self.k_p = k_p
        self.k_d_bo = k_d_bo
        self.k_p_bo = k_p_bo
        self.pi = pi
        self.g = g
        self.n_iterations = n_iterations
        self.n_simulation = n_simulation
        self.x0 = x0
