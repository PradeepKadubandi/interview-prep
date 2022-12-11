import numpy as np
# Partition an array with False and True values
# Should preserve the order of True elements

def swap(nums, i, j):
    tmp = nums[i]
    nums[i] = nums[j]
    nums[j] = tmp

def partition(nums):
    next_true = len(nums)-1
    for i in range(len(nums)-1, -1, -1):
        if nums[i] > 0:
            swap(nums, i, next_true)
            next_true -= 1
    return nums

# To test above, input will be 0's (all False)
# And True's are 1,2,3 etc... will appear in that order...
# Output should preserve the order of input for 1,2,3's...
def test():
    N = 100
    for i in range(N):
        test_length = np.random.randint(50)
        a = np.random.randint(2, size=(test_length))
        j = 1
        for i in range(len(a)):
            if a[i] == 1:
                a[i] = j
                j += 1
        expected = [0 for i in range(test_length-j+1)] + [i for i in range(1, j)]
        actual = partition(a)
        if any(actual != expected):
            print ('Failed test: input: {}, expected: {}, actual: {}'.format(a, expected, actual))
    print ('done')

test()