import math
import matplotlib.pyplot as plt
import numpy as np
import random
import imageio.v2 as imageio

# This class contains methods and a constructor necessary to create a sample perceptron algorithm.
class Perceptron:

    # This constructor enables the creation of a Perceptron object with 2 weights and a bias
    def __init__(self, weight1, weight2, bias, learning_rate):
        self.weight1 = weight1
        self.weight2 = weight2
        self.bias = bias
        self.learning_rate = learning_rate
    
    '''
    This method implements the sigmoid function, which is used by the cost_function and 
    multi_cost_function methods to print values ranging from 0 to 1.
    '''
    def sigmoid(self, x):
        return 1 / (1 + (np.e ** -(x)))
    
    '''
    This method uses the sigmoid function as well as a point (x and y coordinate), weights, and the bias to
    display the 
    '''
    def predict(self, x1, x2):
        return self.sigmoid((self.weight1 * x1) + (self.weight2 * x2) + self.bias)
    


   # This method provides the equation representing the line (decision boundary).
    def yfunction(self, x, a, b, c):
        return (-(a/b) * x) - (c / b)

    # This method creates a graph with points and the decision boundary.
    def graph(self, points, name_string, iteration, xlabel, ylabel):

        '''
        Lists of x and y points respectively. Will help in making the window in the 
        perceptron animation fixed.
        '''
        x_points = []
        y_points = []

        # for point in points:
        #     x_points.append(point[0])
        #     y_points.append(point[1])




        fig = plt.figure()

        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        labels = [p[3] for p in points]

        '''
        Maximum and minimum values for the lists of x points and y points respectively. Will
        help in making the window in the perceptron animation fixed.
        '''
        x_axis_min = min(x_points)
        x_axis_max = max(x_points)
        y_axis_min = min(y_points)
        y_axis_max = max(y_points)

        # xList = np.linspace(x_axis_min - 1, x_axis_min + 1, 100)

        xList = np.linspace(-(x_axis_min), x_axis_max + 1, int(x_axis_max * 2))

        yList = self.yfunction(xList, self.weight1, self.weight2, self.bias)

        # print(x_points)
        # print(y_points)

        plt.axis([x_axis_min - 1, x_axis_max + 1, y_axis_min - 1, y_axis_max + 1])

        plt.scatter(x_points, y_points, c = labels, cmap = 'bwr')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"UCI Heart Data and Decision Boundary")

        plt.plot(xList, yList, color = 'black')

        plt.savefig(name_string)


    
    # This method prints the cost of one point in the graph.
    def cost_function(self, x1, x2, y):
        return -((y * (math.log(self.predict(x1, x2)) + epsilon)) + ((1 - y) * (math.log(1 - self.predict(x1, x2) + epsilon))))
    

    # This method prints the cost of each individual point in an array of points.
    def multi_cost_function(self, points):
        for point in points:
            print()
            cost = (-((point[2] * math.log(self.predict(point[0], point[1]))) + 
                     ((1 - point[2]) * (math.log(1 - self.predict(point[0], point[1])) + epsilon))))
            
            print(f"Cost for ({point[0]}, {point[1]}): {cost}")

    '''
    This method adjusts the weights and biases of the Perceptron object based
    on the placement of the points.
    '''
    def learning(self, list_of_points):
        list_of_weights = [self.weight1, self.weight2, self.bias]
        for i in range(len(list_of_weights)):
            change_sum = 0
            for p in list_of_points:
                # if (list_of_points.index(p) == 0):
                #     print(f"change summation for {i}")
                change_sum += ((self.predict(p[0], p[1]) - p[3]) * (p[i])) * self.learning_rate
                # print(change_sum)
                if (list_of_points.index(p) == len(list_of_points) - 1):
                    print()
            new_w = list_of_weights[i] - (change_sum / len(list_of_points))
            if (i == 0):
                self.weight1 = new_w
            elif (i == 1):
                self.weight2 = new_w
            else:
                self.bias = new_w


    # This function returns the accuracy of the decision boundary on the list of points.
    def get_accuracy(self, list_of_points):

        # Create a variable to keep track of how many points are classified incorrectly.
        count_wrong = 0

        for p in list_of_points:

            t = self.predict(p[0], p[1])

            if t > 0.5 or t == 0.5:
                t = 1
            
            else:
                t = 0

            if t != p[3]:
                count_wrong = count_wrong + 1
        
        error_rate = count_wrong / len(list_of_points)

        return 1 - error_rate

# Epsilon global variable
epsilon = (1 * 10 ** -15)
