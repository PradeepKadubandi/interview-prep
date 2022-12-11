class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        profit = 0
        m = prices[0]
        for i in range(1, len(prices)):
            if (prices[i] - m) > profit:
                profit = prices[i] - m
            m = min(m, prices[i])
        return profit
        