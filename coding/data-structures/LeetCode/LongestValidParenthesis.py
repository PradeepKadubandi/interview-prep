class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        max_length = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if len(stack) > 0:
                    curr_len = i-stack[-1]
                    if curr_len > max_length:
                        max_length = curr_len
                else:
                    stack.append(i)
        return max_length
        