from collections import deque

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.edge = None
        self.next = None
        self.sLink = None
        self.repr = None

    def __repr__(self):
        if not self.repr:
            self.repr = '[val={}, edge={}, next={}, sLink={}]'.format(self.val,
                self.edge,
                self.next.val if self.next else None,
                self.sLink.val if self.sLink else None)
        return self.repr

def bfs(root):
    q = deque()
    q.append(root)
    res = []
    print(root)
    while len(q) > 0:
        line = ''
        while len(q) > 0:
            nextQ = deque()
            n = q.popleft()
            line += str(n)
            line += '    '
            res.append(n)
            if n.next:
                nextQ.append(n.next)
        print (line)
        q = nextQ
    return res

def buildSuffixTrie(A):
    root = TreeNode('')
    leaves = []
    for c in A:
        newLeaf = TreeNode(c)
        curr = newLeaf
        for i in range(len(leaves)-1, -1, -1):
            l = leaves[i]
            l.edge = c
            l.next = TreeNode(l.val + c)
            l.next.sLink = curr
            leaves[i] = l.next
            curr = l.next
        leaves.append(newLeaf)
    return root

bfs(buildSuffixTrie('ABC'))