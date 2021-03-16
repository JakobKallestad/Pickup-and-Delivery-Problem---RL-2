import multiprocessing as mp
from time import time
from functools import partial

def sum_of_cubes(f, n):
    return sum(x**3 for x in range(n))

BIG = 10000

tic = time()
results = []
for i in range(BIG):
    results.append(sum_of_cubes(0, i))
toc = time()
print(toc - tic)


tic = time()
func = partial(sum_of_cubes, 4)
with mp.Pool() as pool:

    results = pool.map(func, range(BIG))
toc = time()
print(toc - tic)

