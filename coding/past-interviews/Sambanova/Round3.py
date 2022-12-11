# Input k, n
# return all possible combinations of length 'k' 
#    from numbers [1,9] without repetetion that yield sum 'n'.


def combinationSum(K, N):
    sol = {}
    for j in range(N+1):
        sol[(1, j)] = [[j]]
    for i in range(2, K+1):
        for j in range(1, N+1):
            val = []
            for k in range(1, j):
                r = sol[(i-1, k)]
                for e in r:
                    if j-k > e[-1]:
                        val.append(e + [j-k])
            sol[(i,j)] = val
    return sol[(K,N)]

result = combinationSum(3, 7)
print (result)