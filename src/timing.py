from time import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def timeit(f):
    def wrapper(*args, **kwargs):
        start = time()
        res = f(*args, **kwargs)
        print("--[%s] took %.2f seconds to run.\n" % 
            (f.__name__, time() - start))
        return res
    return wrapper