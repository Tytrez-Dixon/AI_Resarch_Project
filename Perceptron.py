import math
import matplotlib.pyplot as plt
import numpy as np
import random

# This class contains methods and a constructor necessary to create a sample perceptron algorithm.
class Perceptron:

    # This constructor enables the creation of a Perceptron object with 2 weights and a bias
    def __init__(self, weight1, weight2, bias):
        self.weight1 = weight1
        self.weight2 = weight2
        self.bias = bias
    
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
    def yfunction(self, a, x, b, c):
        return (-(a/b) * x) - (c / b)

    # This method creates a graph with points and the decision boundary.
    def graph(self, points, name_string):

        xList = np.linspace(-6, 10, 20)

        yList = self.yfunction(xList, self.weight1, self.weight2, self.bias)

        fig = plt.figure()

        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        labels = [p[2] for p in points]

        print(x_points)
        print(y_points)

        plt.scatter(x_points, y_points, c = labels)

        plt.xlabel("x1")
        plt.ylabel("x2")

        plt.plot(xList, yList)

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
                if (list_of_points.index(p) == 0):
                    print(f"change summation for {i}")
                change_sum += (self.predict(p[0], p[1]) - p[3]) * (p[i])
                print(change_sum)
                if (list_of_points.index(p) == len(list_of_points) - 1):
                    print()
            new_w = list_of_weights[i] - (change_sum / len(list_of_points))
            if (i == 0):
                self.weight1 = new_w
            elif (i == 1):
                self.weight2 = new_w
            else:
                self.bias = new_w


# Epsilon global variable
epsilon = (1 * 10 ** -15)

# Create perceptron object.
sample_perceptron = Perceptron(random.random(), random.random(), random.random())
print()

print(sample_perceptron.weight1)
print(sample_perceptron.weight2)
print(sample_perceptron.bias)
print()


test_points = [[6, 1, 0], [7, 3, 0], [8, 2, 0], [9, 0, 0], [8, 4, 1], [8, 6, 1],
               [9, 2, 1], [9, 5, 1]]

test_points2 = [[6, 1, 1, 0], [7, 3, 1, 0], [8, 2, 1, 0], [9, 0, 1, 0], [8, 4, 1, 1], [8, 6, 1, 1],
                [9, 2, 1, 1], [9, 5, 1, 1]]



# Test cost_function function
# Test successful
# print(sample_perceptron.cost_function(0, 1, 1))
# print()

# Test multi_cost_function function
# Test successful

print(sample_perceptron.multi_cost_function(test_points))
print()

# Test graph function
# Test successful
sample_perceptron.graph(test_points, "Original_Perceptron_Graph.png")
print()

# Test backprop
# Test successful
for i in range(100):
    sample_perceptron.learning(test_points2)
    sample_perceptron.graph(test_points, "Perceptron_Graph" + str(i) + ".png")




# Print new Perceptron attributes.
print(sample_perceptron.weight1)
print(sample_perceptron.weight2)
print(sample_perceptron.bias)
