'''
Question 2. Given an infinite stream of positive integers ("s"), 
and a target positive integer ("t"), 
implement a function that iterates through the "s" and 
returns when and only when a combination of the numbers seen so far 
sums to "t"

Question 2:
Example 1:
stream = [3, 8, 15, 8, 15, 18, 14, 22, 10, 21, ...]
target = 33
returns at the 2nd 15, because 3 + 15 + 15 = 33

Example 2:
stream = [2, 2, 2, ..., 1, ...] = (a stream consisting of 10 trillion 2's followed by a 1)
target = 5
returns at the 1, because 2 + 2 + 1 = 5
'''

# Observation 1:
# Any approach that stores the input stream as it reads elements
# is intractable given the stream is infinite.

# Observation 2:
# All the numbers from input stream and the target are positive integers.
# So any number higher than target in input stream cannot be in any possible
# combination with sum = target. Also any combination sums higher than
# target are not relevant for solution.

# Assumption 1:
# I am assuming that 'target' is a number that can fit in computer memory.
# In other words, using O(target) space is feasible.
# And the input stream is such that if target sum is achived with
# some combination after n-th element of input stream,
# O(n*target) time is a feasible solution (not intractable for computer).

# Solution approach:
# Keep a set of all possible combination sums that are achievable. 
# When we see the next element from the infinite stream (input sequence):
#   If the number is higher than target, discard it, no need to do anything.
#   Add the new element to all existing possible sums to generate
#   new possible sums (only when they are less than target).
#   Merge the new possible sums into existing possible sums.
# One interesting implementation details is that 
#   We need to do the above in 2 passes instead of one pass because,
#   we don't want to count existing element more than once when generating new possible sums.

# Time and Space complexity
#   Since we don't store any sums higher than 'target', 
#   we store at most 'target' number of elements in the set.
#   So space complexity = O(target)
#   At each element from input, we at most process 'target' number of existing
#      possible sums. So one iteration processing one input element takes O(target) time.
#   So the total time complexity would be O(n*target)
#      where 'n' is the index in input at which we found the target sum.

def hasCombinationSum(it, target):
    possible_sums = set()
    for i, elem in enumerate(it):
        if elem > target:
            continue
        if elem == target:
            # minor optimization, though code below
            # handles this if case, no need to walk through the set in this case.
            return elem, i
        new_sums = set([elem])
        for existingSum in possible_sums:
            if existingSum + elem <= target:
                new_sums.add(existingSum + elem)
        for new_sum in new_sums:
            possible_sums.add(new_sum)
        if target in possible_sums:
            return elem, i

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

    result, actualIndex = hasCombinationSum(iterator, target)
    if actualIndex != expectedIndex:
        print ('Test {} failed, expected index = {}, actual index = {}'.format(i, expectedIndex, actualIndex))
        testFailure = True
    assert result == stream[expectedIndex]

if not testFailure:
    print ('All tests passed!')

