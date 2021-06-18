import functools
import pdb
import sys
import time
from inspect import signature

#Timing
#
#Definition = Is a decorator class that stores execution time of decorated fuctions
#Attributes:
#	+count (int) = class attribute. Is a counter to uniquely identify each function
#       +time_dict (Dict) = class attribute. Each entry is the execution of a function. The pair is key: '{count}-{name_of_function}', value: execution time.
#       +func (Function) = Decorated function.
#       +owner (object) = Is the 'Timing' class.
#       +instance (Timing) = Each instance of the class 'Timing'.
#Methods:
#	+__call__( *args, **kwargs) = Method executed when the decorated function is executed. The arguments corresponds to the arguments of the decorated function.
#       +__get__(instance, owner) = Executed before __call__. Allows each Timing instance to access the instance from which each function is called.
#       +reset__dict() = class method. Reset values of class attributes.
class Timing(object):
    time_dict = dict()
    count = 0
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.owner = None
        self.instance = None

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        if self.instance != None:
            returned = self.func.__call__(self.instance, *args, **kwargs)
        else:
            returned = self.func(*args, **kwargs)
        t1 = time.perf_counter()
        self.time_dict["{:d}-{}".format(self.count, self.func.__name__)] = t1-t0
        self.__class__.count += 1
        return returned

    def __get__(self, instance, owner):
        self.owner = owner
        self.instance = instance
        return self.__call__

    @classmethod
    def reset_dict(cls):
        cls.time_dict.clear()
        cls.count = 0

