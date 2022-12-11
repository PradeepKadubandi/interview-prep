class Solution:
    def search(self, nums, target):
        l = 0
        r = len(nums)-1
        while l <= r:
            mid = int ((l+r) / 2)
            if nums[mid] == target:
                return mid
            if nums[mid] > nums[l]:
                if nums[mid] > target:
                    r = mid-1
                else:
                    l = mid+1
            else:
                if nums[mid] < target:
                    l = mid+1
                else:
                    r = mid-1
        return -1

print (Solution().search([4,5,6,7,0,1,2], 0))