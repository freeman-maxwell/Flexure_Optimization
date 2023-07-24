from fem import get_force_disp_curve, in2m
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')
from scipy import interpolate
from scipy.optimize import minimize

def optimize_this(pts, ani=False):
    pts = in2m(pts)
    u, F = get_force_disp_curve(pts, 10E7, 30, ani=ani)

    # Perform cubic spline interpolation
    spline = interpolate.CubicSpline(u, F)
   
    # Calculate the second derivative using the derivative method of the spline
    second_derivatives = spline.derivative(2)(u[1:-1])
    print("input:", pts)
    print("output:", max(second_derivatives))

    # Find the maximum second derivative and its corresponding data point
    return 1 / max(second_derivatives) 

def opt():
    initial_guess = [0.0, 0.0, 0.0]
    bounds = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))

    # Perform the optimization
    result = minimize(optimize_this, initial_guess, method='L-BFGS-B', bounds=bounds, callback=callback)  # You can try different methods

    # Extract the minimum value and the corresponding input
    minimum_value = result.fun
    minimum_inputs = result.x

    print("Minimum value:", minimum_value)
    print("Inputs at minimum value:", minimum_inputs)

pts = [0, 0, 0, 0.5, 0.5]
optimize_this(pts, ani=True)
