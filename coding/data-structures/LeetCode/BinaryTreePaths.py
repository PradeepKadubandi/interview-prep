# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathFromList(self, l):
        if l is None or len(l) == 0:
            return ''
        pt = str(l[0])
        for i in range(1, len(l)):
            pt += '->'
            pt += str(l[i])
        return pt
    
    def dfsRecursive(self, n, pt, res):
        pt.append(n.val)
        if n.left == None and n.right == None:
            res.append(self.pathFromList(pt))
        else:
            if n.left:
                self.dfsRecursive(n.left, pt, res)
            if n.right:
                self.dfsRecursive(n.right, pt, res)
        pt.pop()
        
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        res = []
        pt = []
        if root != None:
            self.dfsRecursive(root, pt, res)
        return res
        
            
        