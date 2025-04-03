# This file contains sample arguments for running the Perceptron.py file.

# Learning rate global variable
# learning_rate = 0.5

initial_learning_rate = 1

# Global decay parameter variable
decay_parameter = 0.00001

# Global interations variable.
iterations = 10

# Create perceptron object.
sample_perceptron = Perceptron(random.random(), random.random(), random.random(), 1)
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
sample_perceptron.graph(test_points2, "Original_Perceptron_Graph.png", 0)
print()

list_of_file_names =[]

'''
Global parameter for learning_decay function. Keep track of the amount
of times the learning rate changes.
'''
learning_step = 0




# Test backprop
# Test successful
for i in range(iterations):
    print(f"Iteration: {i}")
    print(f"Learning Rate: {sample_perceptron.learning_rate}")
    sample_perceptron.learning(test_points2)
    learning_step += 1
    sample_perceptron.learning_rate = ((sample_perceptron.learning_rate) / (1 + decay_parameter*i))
    sample_perceptron.graph(test_points2, "Perceptron_Graph" + str(i) + ".png", i)
    list_of_file_names.append(f"Perceptron_Graph" + str(i) + ".png")


ims = [imageio.imread(f) for f in list_of_file_names]

imageio.mimwrite("Perceptron.gif", ims)




# Print new Perceptron attributes.
print(sample_perceptron.weight1)
print(sample_perceptron.weight2)
print(sample_perceptron.bias)
