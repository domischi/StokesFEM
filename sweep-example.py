from example import ex
from tqdm import tqdm
import numpy as np

tmp1 = np.linspace(.1,1.,5)
tmp2 = np.linspace(.9,1,11)
ARs=np.union1d(tmp1,tmp2)[::-1]
del tmp1, tmp2

for ar in tqdm(ARs):
    config_updates = {'AR': ar}
    assert(config_updates['AR']>0)
    ex.run(config_updates=config_updates)
