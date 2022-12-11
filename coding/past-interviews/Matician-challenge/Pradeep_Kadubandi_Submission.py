import sys
import numpy as np
import numba
from numba import jit
import timeit
from timeit import Timer

'''
---------------------------------
Running the code:
---------------------------------
  - Create a pipenv environment (tested with python 3.7) with the dependencies below installed and run this script in that environment shell.
  - Dependencies:
    - numpy 
    - numba
'''

'''
---------------------------------
Some general notes/assumptions:
---------------------------------
  - For the input 2-D array, I used np.ubyte (with range 0 to 255) [Overview.2]
  - Since the result can have integers ranging from -255 to 255 [Requirements.2], I used np.int16 as the destination type for convolutions.
  - I assumed the convolution to match the shape of output to input (padding='same' option in most libraries) [Requirements.7].
  - In general, I took the exercise as aiming for a better Time complexity in implementation and memory usage.
    There is a separate pivot to speed up performance by parallel computation and/or using GPU hardware, I did not pay too
    much attention along those aspects in the interest of time (they are two different axes for achieving speed).
  - Further Thoughts about parallelization:
    Convolutions are embrassingly parallelizable when each target element is calculated in parallel (since the input array is read only).
    Min max is also parallelization though care must be taken to update the running min and max variables as that's shared state among 
    parallel executions of different input elements.
    These are interesting avenues for further improvements. 
'''

'''
---------------------------------
Interesting notes / assumptions:
---------------------------------
  - [Requirements.3 - more like hint than requirement] Can assume fixed filter. 
    - The implementation of convolutions is purpose built implementation for the specific filter in question.
      - Basically create the target array by additions (1) / subtractions (-1) of input array elements positioned according to filter.
      - I tried two different implementations following this logic 
        - 1) using default numpy slicing
        - 2) using array iteration in natural order and optimize the code with numba jit compilation to speed up execution.
    - The advantage of above methods (compared to methods that work for any filter) is speed in execution (with creative implementation)
    - The obvious disadvantage is that the implementation is not general.

  - [Overview.6]: Print total time taken by the machine in computing Dx and Dy matrices:
    - Some background Reference (motivation for using timeit): https://www.oreilly.com/library/view/python-cookbook/0596001673/ch17.html
    - Though the statement sounds like measure the time taken to execute one call to convolutions, 
      I took the real need behind this as to compare different implementations to judge which one performs fast.
      With this assumption, I used timeit module to measure repetitions of code execution 
      (using the Timer.autorange method to automatically decide the no. of repetitions).
      While printing, I am printing the average (mean) time taken per repetition, arguably there are other
      ways to report the time taken for a single call (which are not possible to do with values reported from autorange).
    - Since I used timeit to measure multiple calls, I have another explicit function call before timeit to actually
      execute the function and save the result in a variable (There are ways to avoid this with some implications).
      This also means that numba functions are jitted in that first call and compilation overhead is not considered
      in timeit measurement (not too important because we are anyway using multiple calls to measure time).

  - [Overview.7]: Compute the min and max values for both Dx & Dy matrices individually...
    - I took this as min and max for a given 2-D matrix can be computed in one function call and I need to 
      call such function separately for inputs Dx and Dy to get the min and max for each.
    - If min and max needs to be calculated separately, perhaps np.min and np.max are good enough implementations.
      Intuitively, calculating min and max in one function call could avoid iterating over array twice to find min
      and max separately and would be faster. There is another nice optimization to take two values from input
      array and compare lower of them with running min and higher of them with running max to calculate min and max.
      This reduces the number of comparisons to 3/4 folds and should be faster for large arrays (better in terms of growth of time with input).
    - I added these two methods in the script : numba_minmax_with_optimized_comparisons is the version with
      less comparisons. minmax_with_numpy_library is just calling numpy libraries in 2 separate calls.
    - I did a detailed performance comparison of the two methods in the jupter notebook and numba method
      clearly wins! I haven't added the details in here as it was not asked in requirements but can
      share the details later if I get a chance.
'''

'''
---------------------------------
Start: Horizontal convolution implementations. (Only two among what I tried are included here).
---------------------------------
Note1: 
On a separate notebook, I tried the library methods out of curiosity (to measure relative times).
(Intuitively I guessed that purpose built implementation will outperform library implementations).
I tried pytorch, numpy.convolve and scipy.convolve and those methods performed worse w.r.t time
than the purpose built implementation as expected.
Obviously those methods are more generic in implementation and can work for any filter.

Note2:
Using FFT methods for convolutions seemed like a promising avenue to research further. However, intuitively
FFT is still doing fast multiplications reducing linear complexity to logarithmic complexity but
the purpose built algorithm converts the problem to simple additions and avoids multiplications altogether.
Hence I did not delve much into application of FFT based methods to solve this specific problem in
the interest of time.
'''

# I observed that using numba on these implementations which use numpy slicing make them perform worse.
def Dx_with_filter_specific_custom_impl(A):
    '''
    Assumes A is a 2D ndarray. (True for all other methods as  well)
    '''
    Dx = np.zeros_like(A, dtype=np.int16)
    Dx[:, 1:] -= A[:, :-1]
    Dx[:, :-1] += A[:, 1:]
    return Dx

# This method performs poorly if not using numba!
@jit(nopython=True)
def numba_Dx_with_filter_specific_custom_impl(A):
    Dx = np.zeros_like(A, dtype=np.int16)
    R, C = A.shape
    # Changing the order of loops doesn't impact functionality but reduces the performance because of accessing out of natural memory layout of array!!!
    for i in range(R):
        for j in range(C):
            if j > 0:
                Dx[i,j] -= A[i,j-1]
            if j < C-1:
                Dx[i,j] += A[i,j+1]
    return Dx

'''
---------------------------------
Start: Vertical convolution implementations similar to above methods.
---------------------------------
'''
def Dy_with_filter_specific_custom_impl(A):
    '''
    This implemention performs poorer than similar horizontal convolution
    implementation because the slicing is not according to natural memory layuot of array.
    '''
    Dy = np.zeros_like(A, dtype=np.int16)
    Dy[1:, :] -= A[:-1, :]
    Dy[:-1, :] += A[1:, :]
    return Dy

@jit(nopython=True)
def numba_Dy_with_filter_specific_custom_impl(A):
    Dy = np.zeros_like(A, dtype=np.int16)
    R, C = A.shape
    # Changing the order of loops doesn't impact functionality but reduces the performance because of accessing out of natural memory layout of array!!!
    for i in range(R):
        for j in range(C):
            if i > 0:
                Dy[i,j] -= A[i-1,j]
            if i < R-1:
                Dy[i,j] += A[i+1,j]
    return Dy

'''
---------------------------------
Start: Min max calculation implementations.
---------------------------------
'''
@jit(nopython=True)
def numba_minmax_with_optimized_comparisons(A):
    '''
    One potential optimization that can further be done which is very specific
    to the problem is to compare the running min and max with limits on min and max
    values (-255 and 255) and stop the iteration when both min and max
    reach the limits. It adds 2 additional comparisons in the inner loop but could
    potentially end the loop faster (depending on the input array though).
    It seemed too hacky to the specific problem, and when I tested it briefly,
    it did not improve performance (even for large array dims > 1024,1024).
    So did not include it in the final code.
    '''
    A = A.ravel()
    n = A.size
    odd = n % 2
    if not odd:
        n -= 1
    max_val = min_val = A[0]
    i = 1
    while i < n:
        x = A[i]
        y = A[i + 1]
        if x > y:
            x, y = y, x
        min_val = min(x, min_val)
        max_val = max(y, max_val)
        i += 2
    if not odd:
        x = A[n]
        min_val = min(x, min_val)
        max_val = max(x, max_val)
    return min_val,max_val

# This method grows faster with input size, so performs poorly at higher input sizes.
def minmax_with_numpy_library(A):
    return np.min(A), np.max(A)

'''
---------------------------------
Placeholder functions 
---------------------------------
To abstract the implementation of operations from main method of the script.
Useful to invoke different implementations for testing (without changing the main method).
I did this just to alternate between options in this file, when I tried
things in Jupyter, I used lambdas to represent different implementations and 
used list of lambdas to invoke and measure different implementations which is a better way to do this.
'''
def horizontal_convolve(A):
    return numba_Dx_with_filter_specific_custom_impl(A) 

def vertical_convolve(A):
    return numba_Dy_with_filter_specific_custom_impl(A)

def min_max(A):
    return numba_minmax_with_optimized_comparisons(A)

'''
---------------------------------
Main method (outline of requirements)
---------------------------------
'''
def main(rows, cols):
    A = np.random.randint(0, 256, size=(rows, cols), dtype=np.ubyte)
    Dx = horizontal_convolve(A)
    Dy = vertical_convolve(A)
    t = Timer(lambda: horizontal_convolve(A))
    n_calls, total_time = t.autorange()
    print ('Average Time for horizontal convolution = {} sec, averaged over {} calls'.format(total_time / n_calls, n_calls))
    t = Timer(lambda: vertical_convolve(A))
    n_calls, total_time = t.autorange()
    print ('Average Time for vertical convolution = {} sec, averaged over {} calls'.format(total_time / n_calls, n_calls))
    min_val, max_val = min_max(Dx)
    print ('Dx: min = {}, max = {}'.format(min_val, max_val))
    min_val, max_val = min_max(Dy)
    print ('Dy: min = {}, max = {}'.format(min_val, max_val))
    # In case, we would like to see time for min max calculations.
    # t = Timer(lambda: min_max(Dx))
    # n_calls, total_time = t.autorange()
    # print ('[Extra:] Average Time for min max calculation = {} sec, averaged over {} calls'.format(total_time / n_calls, n_calls)) # Not asked in requirements, added for curiosity.


if __name__ == "__main__":
    if len(sys.argv) != 3 or any(filter(lambda s:not s.isdecimal(), sys.argv[1:])):
        print ('Usage : {} <#rows> <#cols>'.format(__file__))
    else:
        main(*map(int, sys.argv[1:]))