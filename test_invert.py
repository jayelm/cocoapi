from pycocotools import mask as cmask
import numpy as np
import numpy as nnp
from tqdm import trange

assert cmask.invert

for _ in trange(10000):
    h = np.random.randint(0, 1000)
    w = np.random.randint(0, 1000)
    prop = np.random.rand()
    x = (np.random.random((h, w)) > prop).astype(np.uint8)
    notx = np.logical_not(x).astype(np.uint8)

    xm = cmask.encode(np.asfortranarray(x))

    notxm = cmask.invert(xm)

    assert (notx == cmask.decode(notxm)).all()
