# O(n) time-complexity solution that calculates average of current window
# using average of previous window and only the 2 edge elements one of which
# was removed from the window and one which was added.
# (Instead of calculating new average by summing all elements again which
# will result in total run time complexity of O(n^2))
def movingAverage(arr, k):
    '''
    Assumption: arr is a list of floating points 
    Assumption: k is an integer
    '''
    if k <= 0:
        raise Exception("Invalid Argument")
    result = []
    if len(arr) < k:
        return result
    current_average = sum(arr[:k]) / k
    result.append(current_average)
    for i in range(k, len(arr)):
        current_average += (arr[i] - arr[i-k]) / k
        result.append(current_average)
    return result

# Each test case is a tuple with first two as arguments to call the above method
# and third value in tuple expected answer from the method.

# no test included for k being zero or negative
testCases = [
    ( [], 1, [] ), #trivial case: len(arr) = 0 < window_length
    ( [1], 2, [] ), #trivial case: 0 < len(arr) < window_length
    ( [1, 3, 5.5], 1, [1, 3, 5.5] ), # window_length = 1 test case
    ( [1, 2, 3, 4, 5.75], 3, [2, 3, 4.25] ), # window_length > 1 test case
]

testFailure = False
for test in testCases:
    arr, k, expected = test
    actual = movingAverage(arr, k)
    if expected != actual:
        print ('Failed test: arr = {}, k = {}, expected = {}, actual = {}'.format(arr, k, expected, actual))
        testFailure = True

if not testFailure:
    print ('All tests passed!')
