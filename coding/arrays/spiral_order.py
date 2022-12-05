# Collect elements of a 2-D array in spiral order
import numpy as np

def spiral_order(a: np.ndarray):
    assert a.ndim == 2 
    result = list()

    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    current_dir = 0
    current_idx = (0, 0)
    added = np.zeros_like(a)

    def add_tuples(t1, t2):
        return tuple(a+b for a,b in zip(t1, t2))

    for i in range(len(a) * len(a[0])):
        result.append(a[*current_idx])
        added[*current_idx] = 1
        next_idx = add_tuples(current_idx, dirs[current_dir])
        if next_idx[0] >= len(a) or next_idx[0] < 0 or next_idx[1] >= len(a[0]) or next_idx[1] < 0 or added[*next_idx] == 1:
            current_dir = (current_dir + 1) % 4
            next_idx = add_tuples(current_idx, dirs[current_dir])
        current_idx = next_idx

    # left, right, top, bottom = 0, len(a[0])-1, 0, len(a)-1
    # while left <= right or top <= bottom:
    #     result.extend(a[top, left:right]) # n-1 elements of top row
    #     result.extend(a[top:bottom, right]) # n-1 elements of right column
    #     result.extend(a[bottom, right:left:-1]) # n-1 elements of bottom row in reverse
    #     result.extend(a[bottom:top:-1, left]) # n-1 elements of left column in reverse

    #     top += 1
    #     right -= 1
    #     bottom -= 1
    #     left += 1

    return result

def main():
    tests = [
        # # single element
        # (
        #     [[1]], [1] 
        # ),
        # # single row
        # (
        #     [[1, 2]], [1, 2] 
        # ),
        # (
        #     [[1, 2, 3, 4]], [1, 2, 3, 4] 
        # ),
        # # single col
        # (
        #     [[1], [2]], [1, 2]
        # ),
        # (
        #     [[1], [2], [3]], [1, 2, 3]
        # ),
        # multi
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 3, 6, 9, 8, 7, 4, 5]
        ),
        (
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [1, 2, 3, 6, 9, 12, 11, 10, 7, 4, 5, 8]
        ),
        (
            [[1, 2, 3, 4], [5, 6, 7, 8]], [1, 2, 3, 4, 8, 7, 6, 5]
        ),
    ]

    tests_failed = False
    for a, expected in tests:
        actual = spiral_order(np.array(a))
        if expected != actual:
            print (f"Test failed: a: {a}, expected: {expected}, actual: {actual}")
            tests_failed = True
    if not tests_failed:
        print ("All tests passed!")

if __name__ == "__main__":
    main()