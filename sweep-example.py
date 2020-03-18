from example import ex
from tqdm import tqdm
import numpy as np

mus = np.logspace(-3,3,3)

for mu in mus:
    config_updates = { 'mu': float(mu)}
    ex.run(config_updates=config_updates)
