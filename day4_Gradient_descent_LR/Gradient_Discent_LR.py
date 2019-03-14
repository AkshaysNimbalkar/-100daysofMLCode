import numpy as np
from numpy import *


# For defining error from each points i.e SSE (Sum of Squared error):
def compute_error_for_given_points(b, m, points):
    totalError = 0
    # need to compute for every point
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    # gradient Discent
    b_grident = 0
    m_grident = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_grident += -(2/N) * (y - ((m_current * x) + b_current))
        m_grident += -(2/N) * x * (y - ((m_current * x) + b_current))
    # gradient is value we use to update values of b and m
    # learning rate tells us how fast we want our model to execute
    new_b = b_current - (learning_rate * b_grident)
    new_m = m_current - (learning_rate * m_grident)
    return [new_b, new_m]


def gradient_descent_runner(points, learning_rate, starting_b, starting_m, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]



def run():
    points = genfromtxt('E:/DataScience_Study/3months/Data-Lit/week4-Regression/linear_regression_live-master/data.csv', delimiter=',')

    # hyperparameters
    learning_rate = 0.0001

    # y= mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000 # bcoz our data set is small
    [b , m] = gradient_descent_runner(points, learning_rate, initial_b, initial_m, num_iterations)

if __name__ == '__main__':
    run()
