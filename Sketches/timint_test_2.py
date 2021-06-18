from timing import Timing

@Timing
def func3(val):
    count = 0
    for i in range(val):
        count += 1
    return count

