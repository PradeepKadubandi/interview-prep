# Multiple approaches are possible to solve this problem:

# Approach 1:
#    Build a set of sum of all possible pairs as we walk through input
#    So, when we look at element 'e' at 'k'-th index:
#       if (target-e) is in the above set, return
#       else visit all the elements seen before and calculate pair sum with 'e' and add that to the set.
#  TIME and SPACE complexity?
#    Processing only 'k'-th element takes O(k) time.
#    So the total runtime is O(n^2) where 'n' would be the returned index.
#    Since we are storing all the 'n' numbers and also 'n(n-1)/2' possible pair-sums,
#         the space complexity of the approach is also O(n^2)

# Approach 2:
#    Instead of building a set, maintain a sorted list of numbers seen so far 
#       and use two pointer approach to see if target sum is achievable.
#    So, when we look at element 'e' at 'k'-th index:
#       We have all numbers from stream[0:k] in sorted order available,
#       search for pairwise sum = (target-e) in the above sorted list by running two indices one from left and one from right.
#       If we didn't find the above sum, then insert 'e' into the sorted list of numbers to maintain the property. (This takes an additional O(k) processing)
#  TIME and SPACE complexity?
#    Here as well, processing only 'k'-th element taken O(k) time.
#    So the total runtime is O(n^2) where 'n' would be the returned index.
#    In this approach, we are only storing 'k' numbers seen so far, so space complexity of O(n).

# Implementing Approach (2) because it has better space complexity and the same time complexity as Approach (1).
# Using bisect api from python to insert a new number into already sorted list.

import itertools
import bisect

def hasPairSum(sortedArr, target):
    '''
    Search for a pairwise sum = 'target' in a given 'sortedArr'
    Time = O(n), Space = O(1), uses 2-pointer approach, does not use set to store membership.
    '''
    l, r = 0, len(sortedArr)-1
    while l < r:
        curr_sum = sortedArr[l] + sortedArr[r]
        if curr_sum == target:
            return True
        elif curr_sum < target:
            l += 1
        else:
            r -= 1
    return False

def waitForTargetSum(it, target):
    '''
    Assumption: it is an iterator over infinite stream of positive integers, so call next to get the next element.
    Assumption: target is a positive integer
    '''
    sorted_list = []
    predicate = lambda elem: not hasPairSum(sorted_list, target-elem)
    for elem in itertools.takewhile(predicate, it):
        bisect.insort(sorted_list, elem)
    return len(sorted_list) # The return value indicates the index at which the method found the result

stream = [3, 8, 15, 8, 15, 18, 14, 22, 10, 21]
iterator = iter(stream)
target = 33
expectedIndex = 4 # corresponds to second 15

testCases = [
    {
        'stream': [3, 8, 15, 8, 15, 18, 14, 22, 10, 21],
        'target': 33,
        'expectedIndex': 4 # corresponds to second 15
    },
    {
        'stream': [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2],
        'target': 5,
        'expectedIndex': 9 # corresponds to 1
    }
]

testFailure = False
for i, test in enumerate(testCases):
    stream = test['stream']
    target = test['target']
    expectedIndex = test['expectedIndex']
    iterator = iter(stream)

    actualIndex = waitForTargetSum(iterator, target)
    if actualIndex != expectedIndex:
        print ('Test {} failed, expected index = {}, actual index = {}'.format(i, expectedIndex, actualIndex))
        testFailure = True

if not testFailure:
    print ('All tests passed!')
