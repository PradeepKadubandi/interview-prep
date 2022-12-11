# Array increasing and then decreasing
# We want to find the peak

# At least 3 elements

# At peak a[i-1] and a[i+1] < a[i]

# [1,3,2]
# [1,2,3,2]
# [1,3,5,4,3,2,1]

# Small inputs (<3)
# Even and odd length
# Invalid case
# Left < right , opposite

def findPeak(a):
    left = 0
    right = len(a)-1
    while left < right: 
        mid = int((left + right) / 2)
        if a[mid] > a[mid-1] and a[mid] > a[mid+1]:
            return a[mid]
        elif a[mid] < a[mid-1]:
            right = mid
        else:
            left = mid

tests = {
    1: ([1,3,2], 3),
    2: ([1,2,3,2], 3),
    3: ([1,3,5,4,3,2,1], 5),
    4: ([1,3,5,7,9,11,13,4,3,2,1], 13)
}

for test, tup in tests.items():
    actual = findPeak(tup[0])
    expected = tup[1]
    if actual != expected:
        print ('Failed test: {}, expected = {}, actual ={}'.format(test, expected, actual))