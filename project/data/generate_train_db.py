import numpy as np
from scipy.optimize import minimize

def rosen(a,b,x):
    """The Rosenbrock function"""
    return (sum(a*(x[1:]-x[:-1]**2.0)**2.0 + (b-x[:-1])**2.0))**2

def main():
    x0 = np.array([0, 0])
    data = np.zeros([10000, 4])
    count = 0
    while (count < 10000):
        a = np.random.uniform(-10,10)
        b = np.random.uniform(-10,10)
        def f(x):
            return rosen(a,b,x);
        res = minimize(f, x0, method='nelder-mead',
                        options={'xatol': 1e-8, 'disp': True})
        x1, x2 = res.x
        data[count] = [a,b,x1,x2]
        count += 1
        print(count)
    np.savez('./db.npz', db=data)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise e
