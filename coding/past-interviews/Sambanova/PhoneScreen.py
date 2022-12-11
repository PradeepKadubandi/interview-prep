# aaecbaaaeiouu
# count # of substrings that contain *only* vowels and each vowel appearing at least once
# aaeiouu
# aeiouu
# aeiou
# aaeiou - 4

# Brootforce - exponential
# Divide and conquer: 
# Solve upto T(n-1), C(n-1)
# if a[n] is consonant: C(n) = C(n-1)
# a[n] is vowel : 'a' 1) C(n) = 

vowels = set(['a', 'e', 'i', 'o', 'u'])

def isVowel(c):
    return c in vowels

def totalSubStrings(left, right, lSpan, rSpan):
    '''
    Calculates total substrings within a window of
    characters which are all vowels.
    left: left most index of window in input string
    right: right most index of window in input string
    lSpan: indef of first character from left where the span covering all vowels starts
    rSpan: right index of above span (inclusive)
    '''
    return ((lSpan-left+1) * (right-rSpan+1))

def hasAllVowels(vowelCount):
    for k in vowelCount.keys():
        if vowelCount[k] == -1:
            return False
    return True

def countSubStrings(a):
    if len(a) < 5:
        return 0
    j = 0
    count = 0
    while j < len(a):
        left = j
        vowelCount = {'a':-1, 'e':-1, 'i':-1, 'o':-1, 'u':-1}
        while j < len(a) and isVowel(a[j]):
            vowelCount[a[j]] = j
            if hasAllVowels(vowelCount):
                rSpan = j
                lSpan = min(vowelCount.values())
                while (j < len(a) and isVowel(a[j])):
                    j += 1
                count += totalSubStrings(left, j-1, lSpan, rSpan)
            else:
                j += 1
        j += 1
    return count

tests = {
    'aeiou': 1,
    'iaoue': 1,
    'aeaeioiouuu': 9,
    'aaecbaaaeiouu': 6,
    'bacaeaioou': 2,
    'babababa': 0
}
for test,expected in tests.items():
    allPass = True
    actual = countSubStrings(test)
    if actual != expected:
        print ('Test {} Failed, Expected = {}, Actual = {}'.format(test, expected, actual))
        allPass = False
if allPass:
    print ('All tests passed!')

