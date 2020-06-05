from multiprocessing import Pool



def func(i):
    print(i)
if __name__ == "__main__":
    pool = Pool()
    pool.map(func, list(range(1,100)))