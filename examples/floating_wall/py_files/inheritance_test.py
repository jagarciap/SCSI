class A(object):
    def __init__(self):
        self.value = 'A'
        print("Created A")

    def function(self, func, arg):
        print("Holi, soy A", self.value)
        return func(arg)+1

    def function3(self, arg):
        print("Holi, soy A", self.value)
        return super(A, self).function3(arg)+1

class B(object):
    def __init__(self):
        super().__init__()
        self.value = 'B'
        print("Created B")

    def function(self, arg):
        print("Holi, soy B", self.value)
        return arg+1

    def function3(self, arg, existo = 0):
        print("Holi, soy B", self.value)
        return arg+1

class C(A,B):
    def __init__(self, children, value = 'C'):
        super().__init__()
        super(A, self).__init__()
        self.value = value
        self.children = children
        print("Created C")

    def function(self, arg):
        print("Holi, soy C", self.value)
        return super().function(super(A, self).function, arg)+1

    def function2(self, arg):
        return super(A, self.children[0]).function(arg)


child = C([], value = 'child')
parent = C([child], value = 'parent')
alone = C([])
print(parent.function2(1))
print(parent.function(1))
print(alone.function3(1))
