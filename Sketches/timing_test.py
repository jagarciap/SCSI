from timing import Timing
from timint_test_2 import func3

print("Hi!")

n_dict = {}
@Timing
def func1(val1):
    c = 0
    for i in range(val1):
        c += 1
    return c

@Timing
def func2():
    val = func1(int(1e5))
    for i in range(val,val*2):
        val +=1
    return val

val = func1(100)
print(val)
val = func2()
print(val)
times = getattr(Timing, 'time_dict')
print(times)
setattr(Timing, 'time_dict', {})

print(func3(10000))
print(func3(10001))

@Timing
def func4(val):
    count = 0
    for i in range(val):
        count += 1
    return func3(count)

print(func4(20000))

times = getattr(Timing, 'time_dict')
print(times)
