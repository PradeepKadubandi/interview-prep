'''
Given a time represented in the format "HH:MM", form the next closest time by reusing the current digits. 
There is no limit on how many times a digit can be reused.

You may assume the given input string is always valid. 
For example, "01:34", "12:09" are all valid. "1:34", "12:9" are all invalid.

Input: "19:34"
Output: "19:39"

Input: "23:59"
Output: "22:22"
'''

class Solution(object):
    def nextClosestTime(self, inStr):
        """
        :type time: str
        :rtype: str
        """
        def findDiff(hh1, mm1, hh2, mm2):
            adj_h2, adj_m2 = hh2, mm2
            if hh2 < hh1 or (hh2 == hh1 and mm2 <= mm1):
                adj_h2 += 24
            if mm2 < mm1:
                adj_m2 += 60
                adj_h2 -= 1
            return 60 * (adj_h2 - hh1) + (adj_m2 - mm1)
            # minutes = (mm2 - mm1) - (0 if mm1 <= mm2 else 60)
            # hours = (hh2 - hh1) if (hh1 < hh2 or (hh1 == hh2 and mm1 < mm2)) else (hh2 + (24 - hh1))
            # return 60 * hours + minutes

        def generateTimes(digits, pos, all_values):
            if pos == 4:
                all_values.append([e for e in digits])
                return
            for d in digits:
                old = digits[pos]
                digits[pos] = d
                generateTimes(digits, pos+1, all_values)
                digits[pos] = old
            return all_values

        parts = inStr.split(':')
        hh1, mm1 = int(parts[0]), int(parts[1])
        digits = [int(hh1/10), hh1%10, int(mm1/10), mm1%10]
        min_diff = 24*60
        res = inStr
        all_values = []
        generateTimes(digits, 0, all_values)
        print (len(all_values))
        for seq in all_values:
            hh2, mm2 = seq[0]*10 + seq[1], seq[2]*10 + seq[3]
            diff = findDiff(hh1, mm1,  hh2, mm2)
            if diff < min_diff:
                min_diff = diff
                res = '{}:{}'.format(hh2, mm2)
        return res

def findDiff(hh1, mm1, hh2, mm2):
    adj_h2, adj_m2 = hh2, mm2
    if hh2 < hh1 or (hh2 == hh1 and mm2 <= mm1):
        adj_h2 += 24
    if mm2 < mm1:
        adj_m2 += 60
        adj_h2 -= 1
    return 60 * (adj_h2 - hh1) + (adj_m2 - mm1)
    # minutes = (mm2 - mm1) - (0 if mm1 <= mm2 else 60)
    # hours = (hh2 - hh1) if (hh1 < hh2 or (hh1 == hh2 and mm1 < mm2)) else (hh2 + (24 - hh1))
    # return 60 * hours + minutes

print (findDiff(20, 48, 22, 22))

# sol = Solution()
# print (sol.nextClosestTime('19:34'))