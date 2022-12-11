class Solution:
    def threeSum(self, nums):
        nums = sorted(nums)
        result = set()
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i-1]:
                # # Odd edge case: equal to or more than 3 zeros in input
                # if nums[i-1] == 0:
                #     if i+1 < len(nums) and nums[i+1] == 0:
                #         if i == 1 or (i > 1 and nums[i-2]) != 0:
                #             result.append([0, 0, 0])
                continue
            if nums[i] > 0:
                break
            l = i+1
            r = len(nums)-1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s > 0:
                    r -= 1
                elif s < 0:
                    l += 1
                else:
                    result.add((nums[i], nums[l], nums[r]))
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l-1]:
                        l += 1
                    while r > l and nums[r] == nums[r+1]:
                        r -= 1
        return list(result)

print (Solution().threeSum([1,1,-2]))