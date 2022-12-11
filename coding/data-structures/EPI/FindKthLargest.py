import random
import numpy as np

def swap(a, idx1, idx2):
    tmp = a[idx1]
    a[idx1] = a[idx2]
    a[idx2] = tmp

def partition(a, left, right, pivot_idx):
    swap(a, right, pivot_idx)
    new_pivot_idx = left
    for i in range(left, right):
        if a[i] < a[right]:
            swap(a, i, new_pivot_idx)
            new_pivot_idx += 1
    swap(a, new_pivot_idx, right)
    return new_pivot_idx

def findKthLargest(a, k):
    if len(a) < k:
        return None
    left = 0
    right = len(a)-1
    while left <= right:
        pivot_idx = random.randint(left, right)
        new_pivot_idx = partition(a, left, right, pivot_idx)
        if new_pivot_idx == k-1:
            return a[new_pivot_idx]
        elif new_pivot_idx > k-1:
            right = new_pivot_idx-1
        else:
            left = new_pivot_idx+1
    return None

def findBySorting(a, k):
    if len(a) < k:
        return None
    return sorted(a)[k-1]

def generateTest(n):
    for i in range(n):
        l = np.random.randint(0, 101)
        if l == 0:
            return [], 1
        k = np.random.randint(0, l)
        a = np.random.randint(0, 10, size=(l))
        yield a,k+1

# tests = {
#     1:([4,7,1,3,6,5,2], 1),
#     2:([4,7,1,3,6,5,2], 3),
#     3:([4,7,1,3,6,5,2], 5),
#     4:([4,7,1,3,6,5,2], 7),
#     5:([4,7,1,3,6,5,2], 9),
# }

# for tId, test in tests.items():
#     a, k = test
#     actual = findKthLargest(a, k)
#     expected = findBySorting(a, k)
#     if actual != expected:
#         print ('Failed Test: {}, actual = {}, expected = {}'.format(test, actual, expected))

for a,k in generateTest(100):
    # print ('{}, {}, {}'.format(len(a), a,k))
    actual = findKthLargest(a, k)
    expected = findBySorting(a, k)
    if actual != expected:
        print ('Failed Test: {}, actual = {}, expected = {}'.format(test, actual, expected))
print ('Done')
