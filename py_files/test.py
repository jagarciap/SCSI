import numpy

def fun(pos, field):
    array = numpy.repeat(numpy.arange(len(field)), 2)
    numpy.add.at(field, array, pos)

num = 50
field = numpy.zeros((1000))
pos = numpy.ones((num*2))
fun(pos, field[num: num+num])
print(field)
fun(pos, field[num*4: num*4+num])
print(field)
