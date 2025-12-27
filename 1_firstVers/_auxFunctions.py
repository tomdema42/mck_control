import numpy as np




def load_params(filename):
    params = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty lines or comments
            name, value = [part.strip() for part in line.split("=")]
            params[name] = float(value)
    return params

def l1_cost(t, x):
    return np.trapz(x**2, t)
    