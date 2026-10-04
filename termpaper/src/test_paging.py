import numpy as np
from paging import fifo, lru, opt

ref = np.array([7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1])
for name, f, exp in [("FIFO", fifo, 15), ("LRU", lru, 12), ("OPT", opt, 9)]:
    faults = int((~f(ref, 3)).sum())
    assert faults == exp, (name, faults, exp)
    print(f"{name}: {faults} faults  OK")
print("All classical-algorithm tests passed.")
