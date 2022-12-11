# import requests
# import mysql.connector
# import pandas as pd

"""
operations: +, *
no brackets
input: "1+2*3"
expected output: 7
no floats
N >= 0
"""

'''
1+256*3*5+4 : L1 = 0, L2 = 1
1*5+6*4
'''
# Split input string into different parts : array of values
# Process / merge * first
# Add all results 
# L1 : keeps track of the sum of values
# L2 : current multiplication value
# operand_value
# current_operation

# 1+2*3+4 : [1, +, 2, *, 3, +, 4]
# 1+2*3*4

class binary_expr:
    def __init__(self):
        self.operator = None
        self.op1 = None # could be another expr
        self.op2 = None # could be another expr

    def eval(self):
        val1 = int(self.op1) if isinstance(self.op1, str) else self.op1.eval()
        val1 = int(self.op2) if isinstance(self.op2, str) else self.op2.eval()
        return val1 * val2 if self.operator == '*' else val1 + val2
    
    def setoperator(self, op):
        self.op2 = op
    
    def setop1(self, op1):
        self.op1 = op1
    
    def setop2(self, op2):
        self.op2 = op2

# Original method written in the interview
# def evaluate_expression(str):
#     l1 = 0
#     l2 = 1
#     start = 0
#     operators = ['+', '*']
#     is_mul_mode = False
#     while start < len(str):
#         end = start
#         while end < len(str) and str[end] not in operators:
#             end += 1
#         current_operand = int(str[start:end])
#         if end == len(str):
#             if is_mul_mode:
#                 l1 += l2*current_operand
#             else:
#                 l1 += current_operand
#             break
#         if str[end] == '*':
#             l2 *= current_operand
#             is_mul_mode = True
#         elif str[end] == '+':
#             if is_mul_mode:
#                 l1 += l2*current_operand
#                 l2 = 1
#                 is_mul_mode = False
#             else:
#                 l1 += current_operand
#         start = end+1
#     return l1

def parse_expression(str):
    
def evaluate_expression(str):
    l1 = 0
    l2 = 1
    start = 0
    operators = ['+', '*']
    is_mul_mode = False
    while start < len(str):
        end = start
        while end < len(str) and str[end] not in operators:
            end += 1
        current_operand = int(str[start:end])
        if is_mul_mode or str[end] == '*':
            l2 *= current_operand
        if end == len(str):
            if is_mul_mode:
                l1 += l2*current_operand
            else:
                l1 += current_operand
            break
        if str[end] == '*':
            l2 *= current_operand
            is_mul_mode = True
        elif str[end] == '+':
            if is_mul_mode:
                l1 += l2*current_operand
                l2 = 1
                is_mul_mode = False
            else:
                l1 += current_operand
        start = end+1
    return l1
            
        

test_cases = {
    '1+2*3+4': 11,
    '354+2*5*4': 394
}   

for test, expected in test_cases.items():
    actual = evaluate_expression(test)
    print ('Test case : Expr = {}, Expected = {}, Actual = {}'.format(test, expected, actual))

print ('done')