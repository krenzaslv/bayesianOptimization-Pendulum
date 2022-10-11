def integrate(x_t, df, dt=1e-3):
    return x_t + dt * df(x_t)
