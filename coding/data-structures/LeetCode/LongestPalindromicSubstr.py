class Solution:
    def longestPalindrome(self, s: str) -> str:
        '''
        p[i,j] : true if s[i,j] is a palindrome otherwise false
        base:
        p[i,i] = true
        p[i, i+1] = s[i] == s[i+1]
        '''
        if len(s) == 0:
            return ''
        
        # p_prev = [True for i in range(len(s))]
        # p_new = [False for i in range(len(s))]
        result = (0, 1)
        p = [[False for j in range(len(s))] for i in range(len(s))]
        found_longer = False
        for i in range(len(s)):
            p[i][i] = True
            if i+1 < len(s): 
                p[i][i+1] = (s[i] == s[i+1])
                if p[i][i+1] and not found_longer:
                    found_longer = True
                    result = (i, i+2)
                
        for l in range(2, len(s)):
            found_longer = False
            for i in range(len(s)-l):
                p[i][i+l] = p[i+1][i+l-1] and s[i] == s[i+l]
                if p[i][i+l] and not found_longer:
                    found_longer = True
                    result = (i, i+l+1)
                
        # for l in range(len(s)-1, 0, -1):
        #     for i in range(len(s)-l):
        #         if p[i][i+l]:
        #             return s[i:i+l+1]
                
        return s[result[0]:result[1]]
            
            
        