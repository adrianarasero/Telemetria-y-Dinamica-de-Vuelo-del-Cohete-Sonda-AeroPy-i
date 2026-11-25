import numpy as np
from scipy.linalg import solve, lu_factor, lu_solve
from scipy.optimize import bisect, newton as scipy_newton

def my_dot(U:np.ndarray, V:np.ndarray) -> float:
    return sum(U[i] * V[i] for i in range(len(U)))

def my_matmul(A:np.ndarray, B:np.ndarray) -> np.ndarray:
    N, M = np.shape(A)
    return np.array([[my_dot(A[i,:], B[:,j])for j in range (M)] for i in range (N)])

def my_traspose(A:np.ndarray)-> np.ndarray:
    N = np.shape(A)[0] 
    M = np.shape(A)[1]   
    return np.array([[A[j, i]for j in range(N)] for i in range(M)])
# ==========================================
# 1. SISTEMAS LINEALES
# ==========================================
def pivoting_row_swapping(A, b, k, N):
    s = np.zeros(N)
    for i in range(k, N):   
        s[i] = np.max(np.abs(A[i, k:N])) 
    pivot = np.abs(A[k, k] / s[k]) if s[k] != 0 else 0.0
    l = k
    for j in range(k, N): 
        if np.abs(A[j, k] / s[j]) > pivot: 
            pivot = np.abs(A[j, k] / s[j])
            l = j
    if pivot == 0.0:  
        raise ValueError("La matriz es singular")
    if l != k: 
        A[[k, l], k:N] = A[[l, k], k:N]  
        (b[k], b[l]) = (b[l], b[k])

def my_gauss(A_in, b_in):
    """Método de Gauss con pivoteo parcial propio"""
    A = A_in.copy().astype(float)
    b = b_in.copy().astype(float)
    N = len(b)
    x = np.zeros(N) 
    for i in range(N): 
        pivoting_row_swapping(A, b, i, N)
        for k in range(i+1, N):  
            c = A[k, i] / A[i, i]  
            A[k, :] = A[k, :] - c * A[i, :] 
            b[k] = b[k] - c * b[i] 
    for i in range(N-1, -1, -1):
        x[i] = (b[i] - my_dot(A[i, i+1:N], x[i+1:N])) / A[i, i]
    return x

def solver_lu(A, b):
    """Resolución mediante factorización LU"""
    LU, piv = lu_factor(A)
    return lu_solve((LU, piv), b)

def my_jacobi(A, b):
    """Método iterativo de Jacobi"""
    N = len(b)
    x, D, R = np.zeros(N), np.zeros(N), np.zeros((N, N))
    xn = 1 + x 
    tolerance = 1e-10
    k, max_iter = 0, 100
    for i in range(N): 
        for j in range(N): 
            if i == j: 
                D[i] = A[i, j]
                R[i, j] = 0 
            else:
                R[i, j] = A[i, j]
    while np.norm(xn - x) > tolerance and k < max_iter: 
        x = xn
        xn = (b - my_dot(R, x)) / D
        k += 1
    return xn

def my_gauss_seidel(A, b):
    """Método iterativo de Gauss-Seidel"""
    N = len(b)
    x1, x2, b1 = np.zeros(N), np.zeros(N), np.zeros(N)
    Error, tolerance = 1, 1e-10 
    k, max_iter = 0, 100
    while Error > tolerance and k < max_iter: 
        x1[:] = x2[:]  
        for i in range(N):
            b1[i] = b[i] - my_dot(A[i, i+1:], x1[i+1:])  
        for i in range(N):
            x2[i] = (b1[i] - my_dot(A[i, :i], x2[:i])) / A[i, i] 
        Error = np.linalg.norm(x2 - x1)
        k += 1
    return x2


# ==========================================
# 2. REGRESIÓN
# ==========================================
def my_vandermonde_reg(x, N, M): 
    return np.array([[x[j]**i for i in range(N)] for j in range(M)]) 

def my_regression(x, y, degree=1):
    """Regresión por mínimos cuadrados con Vandermonde propio"""
    M = len(x)
    N = degree + 1
    A = my_vandermonde_reg(x, N, M) 
    w = my_gauss(my_dot(A.T, A), my_dot(A.T, y))
    return w  # Coeficientes [c0, c1, ...]

def numpy_regression(x, y, degree=1):
    """Regresión utilizando numpy.polyfit."""
    return np.polyfit(x, y, degree)


# ==========================================
# 3. INTERPOLACIÓN
# ==========================================
def Lagrange_polynomials(x, xp):
    """Polinomios de Lagrange y derivadas """
    N = len(x)-1
    Lagrange = np.zeros((N+1, N+1, N+1)) 
    for j in range(N+1):  
        Lagrange[0, j, 0] = 1 
        n = 1 
        for i in range(N+1):  
            if i != j:   
                for k in range(N, -1, -1): 
                    Lagrange[k, j, n] = (Lagrange[k, j, n-1] * (xp - x[i]) + k * Lagrange[k-1, j, n-1]) / (x[j] - x[i]) 
                n = n + 1 
    return Lagrange[:, :, n-1]

def my_interpolation_lagrange(x, y, xp_val):
    """Interpolación mediante polinomios de Lagrange propios"""
    L = Lagrange_polynomials(x, xp_val)
    return my_dot(L[0, :], y)

def my_interpolation_vandermonde(x, y):
    """Interpolación exacta por matriz de Vandermonde"""
    N = len(x)
    A = np.array([[x[i]**j for j in range(N)] for i in range(N)])
    return my_gauss(A, y)

# ==========================================
# 4. INTEGRACIÓN
# ==========================================
def my_trapecio(y, dx):
    """Regla del trapecio con arrays"""
    return dx / 2 * (y[0] + y[-1] + 2 * np.sum(y[1:-1]))

def my_simpson(y, dx):
    """Regla de Simpson con arrays"""
    N = len(y) - 1
    if N % 2 != 0:
        # Ajuste simple si N no es par
        N = N - 1
    return dx / 3 * (y[0] + y[N] + 4 * np.sum(y[1:N:2]) + 2 * np.sum(y[2:N-1:2]))

def numpy_integration(y, x):
    """Integración con numpy (trapz)."""
    return np.trapz(y, x)
