# Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

# Your algorithm's runtime complexity must be in the order of O(log n).

# If the target is not found in the array, return [-1, -1].# if mid < target: search right

#----------------------------------

# if mid > target: search left
# if mid == target:
      # if mid-1 == target: keep searching left
      # else lowR = mid
     # if mid+1 == target: keep searching right
      # else highR = mid
class Solution:
    def findFirstOccurenceLeft(self, nums, l, r, target):
        '''
        Called when nums[r] == target and potentially the 'target' is to left
        '''
        while l < r:
            mid = int((l+r) / 2)
            if nums[mid] < target:
                l = mid+1
            else:
                r = mid
        return l
    
    def findFirstOccurenceRight(self, nums, l, r, target):
        '''
        called when nums[l] == target and potentially the 'target' is to right
        '''
        while l < r:
            mid = 1 + int((l+r) / 2)
            if nums[mid] == target:
                l = mid
            else:
                r = mid-1
        return l
    
    def searchRange(self, nums, target):
        l, r = 0, len(nums)-1
        lowR, highR = -1, -1
        if r == 0 and nums[0] == target:
            lowR, highR = 0, 0
        while l < r:
            mid = int((l+r) / 2)
            if nums[mid] == target:
                lowR = self.findFirstOccurenceLeft(nums, l, mid, target)
                highR = self.findFirstOccurenceRight(nums, mid, r, target)
                break
            elif nums[mid] < target:
                l = mid+1
            else:
                r = mid-1
        return (lowR, highR)

print (Solution().searchRange([1,4], 4))