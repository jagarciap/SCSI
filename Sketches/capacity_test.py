import numpy
import matplotlib.pyplot as plt

matrix = "inv_capacity_matrix_2020-12-23_17h08m.txt"

def load_file(filename):
    return numpy.loadtxt(filename)

fig = plt.figure(figsize=(12,12))
data = load_file(matrix)
plt.spy(data)
plt.show()
