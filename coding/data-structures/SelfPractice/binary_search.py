'''
Assume duplicates 
'''

def insertBeforeLeftMost(a, key):
    pass

def insertAfterRightMost(a, key):
    pass

def findFirstOccurence(a, key):
    left = 0
    right = len(a)-1
    result = -1
    while left <= right:
        mid = left + int((right - left) / 2)
        if a[mid] > key:
            right = mid-1
        elif a[mid] < key:
            left = mid+1
        else:
            result = mid
            # Nothing to the right of mid can be first occurence
            right = mid-1
    return result

def findRightMostInsertIndex(a, key):
    left = 0
    right = len(a)-1
    while left <= right:
        mid = left + int((right - left) / 2)
        if a[mid] <= key:
            left = mid+1
        else:
            right = mid-1
    return left

nums = [-14, -10, 2, 108, 108, 243, 285, 285, 285, 401]
tests = [(-13, 1), (108, 5), (285, 9)]
for key, expected in tests:
    actual = findRightMostInsertIndex(nums, key)
    print ('Expected = {}, Actual = {}'.format(expected, actual))
