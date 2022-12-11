# def buildPiForPattern(pat):
#     # pi[i] = *length* of longest 'strict'-prefix that's also a suffix
#     # pi[0] = 0 because there are no strict prefixes
#     pi = [0 for c in pat]
#     k = 0
#     for i in range(1, len(pat)):
#         while k > 0 and pat[k] != pat[i]:
#             k = pi[k-1]
#         if k == 0 and pat[i] != pat[0]:
#             pi[i] = 0
#         else:
#             k += 1
#             pi[i] = k
#     return pi

def buildPiForPattern(pat):
    # pi[i] = *index* of end of longest 'strict'-prefix that's also a suffix,
    #         -1 if no such index exists
    # pi[0] = -1 because there are no strict prefixes
    pi = [-1 for c in pat]
    k = -1
    for i in range(1, len(pat)):
        while k >= 0 and pat[k+1] != pat[i]:
            k = pi[k]
        if k == -1 and pat[i] != pat[0]:
            pi[i] = -1
        else:
            k += 1
            pi[i] = k
    return pi

def preprocessPattern(B):
    lps = [0 for _ in range(len(B))]
    i = 1
    j = 0
    while i < len(B):
        if B[i] == B[j]:
            j = j + 1
            lps[i] = j
            i = i + 1
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                lps[j] = 0
                i = i + 1
    return lps

testStr = 'ababcabab'
print (buildPiForPattern(testStr))
print (preprocessPattern(testStr))