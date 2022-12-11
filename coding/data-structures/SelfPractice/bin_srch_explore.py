# Right included / Right not included
# Even / Odd length
# 3 settings for low, high settings

class Interval:
    def __init__(self, low, high):
        self.low = low
        self.high = high
    
def generateTree(L, lowFunc, highFunc):
    root = (0, L)
    thisLevel = [root]
    nextLevel = []
    for i in range(len(thisLevel)):
        node = thisLevel[i]
        mid = node[0] + int((node[1] - node[0]) / 2)
        right = (lowFunc(mid), node[1])
        left = (node[0], highFunc(mid))
        
