from collections import deque

# The key insight missing from this implementation is that
# the dictionary key is just a single character instead of 
# matching prefix of node
class Node:
    def __init__(self, val, children={}, isTerminal=False):
        self.val = val
        self.children = children
        self.isTerminal = isTerminal

    def __repr__(self):
        return 'val = {}, isTerminal = {}, children = {}'.format(self.val, self.isTerminal, self.children.keys)

class Trie:
    def __init__(self):
        self.root = Node('')

    def add(self, val):
        if val == '':
            return
        curr = self.root
        si = 0 #index of new string
        while si < len(val) and val[si] in curr.children:
            curr = curr.children[val[si]]
            i = 0 # index of node
            while si < len(val) and i < len(curr.val) and val[si] == curr.val[i]:
                i += 1
                si += 1
            # if si == len(val): # whole string is matched
            #     if i == len(curr.val):
            #         curr.isTerminal = True
            if i < len(curr.val):
                remString = curr.val[i:]
                newNode = Node(remString, curr.children, isTerminal=curr.isTerminal)
                curr.val = curr.val[:i]
                curr.isTerminal = False
                curr.children = {remString[0]: newNode}
        if si == len(val):
            curr.isTerminal = True
        else:
            newNode = Node(val[si:], isTerminal=True)
            curr.children[val[si]] = newNode

    def search(self, val):
        if val == '':
            return self.root
        curr = self.root
        si = 0
        while si < len(val) and val[si] in curr.children:
            curr = curr.children[val[si]]
            i = 0 # index of node
            while si < len(val) and i < len(curr.val) and val[si] == curr.val[i]:
                i += 1
                si += 1
            if i < len(curr.val):
                return False
        return False if si < len(val) else True

    def print(self):
        thisLevel = [self.root]
        while len(thisLevel) > 0:
            nextLevel = []
            for n in thisLevel:
                print (n)
                for c in n.children.values():
                    nextLevel.append(c)
            thisLevel = nextLevel

t = Trie()
strings = ['abc', 'zeus', 'abstain', 'bow', 'zebra']
for s in strings:
    t.add(s)
t.print()