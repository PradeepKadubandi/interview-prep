# Key part of design is each node has children which is a dictionary
# The dictionary key is just single character
# The values in dictionary need to capture 2 things:
# the value of edge (this is a string and not a single character unlike regular trie)
# the children node (end of edge)
# There are multiple ways to maintain the above 2:
# 1. Could be two separate dictionaries in Node object (This is what Pln's code did)
# 2. A tuple of 2 values (may simplify code but maynot be too readable)
# 3. Node object has 'value' field which captures the value of edge
#    In this design, the child node's value contains the edge value.
# Using the 3rd approach

class TrieNode:
    def __init__(self, val, isWord = False):
        self.val = val
        self.children = {}
        self.isWord = isWord

    def __repr__(self):
        return '({},{}): [{}]'.format(self.val, self.isWord, str.join(',', [str(n) for n in self.children.values()]))

    def add(self, val):
        if len(val) == 0:
            self.isWord = True
            return
        # 2 cases:
        # 1. First letter of val is not in children - strainght forward, add new child
        # 2. First letter has a child - 4 sub cases:
        #    2.a  existingChild is prefix of current val
        #    2.b  val is prefix of existingChild
        #    2.c  val and existingChild share common prefix (atleast first character should match)
        #    2.d  both val and existingChild are the same 
        key = val[0]
        if key in self.children:
            existingNode = self.children[key]
            pre = 0
            while pre < len(val) and pre < len(existingNode.val) and val[pre] == existingNode.val[pre]:
                pre += 1
            if pre < len(existingNode.val):
                part1, part2 = existingNode.val[:pre], existingNode.val[pre:]
                newNode = TrieNode(part2)
                existingNode.val = part1
                newNode.children = existingNode.children
                existingNode.children = {part2[0]: newNode}
                if existingNode.isWord:
                    existingNode.isWord = False
                    newNode.isWord = True
            existingNode.add(val[pre:])
        else:
            self.children[key] = TrieNode(val, isWord=True)

    def search(self, val):
        if val == self.val:
            return self.isWord or self.val == '' # second condition for root and empty string
        if val.startswith(self.val):
            newVal = val[len(self.val):]
            if newVal[0] in self.children:
                return self.children[newVal[0]].search(newVal)
        return False

# I probably don't need this as a class? 
# Just use 'Root' node as Trie object in program 
# (unless new functionality requires a separate abstraction)
class Trie:
    def __init__(self):
        self.root = TrieNode('')

    def add(self, val):
        self.root.add(val)

    def search(self, val):
        return self.root.search(val)

    def __repr__(self):
        return str(self.root)

# ------------------------
# Testing
# ------------------------
t = Trie()
strings = ['abc', 'zeus', 'abstain', 'bow', 'zebra', 'netflix', 'network', 'net', 'next', 'nest']
for s in strings:
    t.add(s)
print(t)

seach_words = {s: True for s in strings}
seach_words[''] = True
for non_word in ['ab', 'ze', 'ne', 'news', 'root', 'absense', 'zedi']:
    seach_words[non_word] = False

for word, expected in seach_words.items():
    actual = t.search(word)
    if expected != actual:
        print ('Search for {} resulted in {} but expected to see {}'.format(word, actual, expected))
else:
    print ('All tests passed') # Exploting a weird python feature :-)

print ('Done')