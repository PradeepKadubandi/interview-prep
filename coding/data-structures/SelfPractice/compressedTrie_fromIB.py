# Implementation from one of Trie problems (under Trees)
# in InterviewBit - but it doesn't seem to be in line with the design
# of compressed trie...
class Node:
    def __init__(self):
        self.data = {}
        
    def __repr__(self):
        return str(self.data)
        
    def add(self, element):
        e, el = element[0], element[1:]
        if e in self.data:
            if type(self.data[e])==str:
                #it's a string
                f, fl = self.data[e][0], self.data[e][1:] 
                self.data[e] = Node()
                self.data[e].data[f] = fl
                self.data[e].add(el)
            else:
                #it's a Node
                self.data[e].add(el)
        else:
            self.data[e]=el
            
    def key(self, value):
        v, val = value[0], value[1:]
        if v in self.data:
            if type(self.data[v])==str:
                return v
            else:
                return v + self.data[v].key(val)
        else:
            return ''

root = Node()
# With this example, the compressed trie should merge nodes 'z' and 'e' into one node
# but the code above does not do that... so implementation is not truly accurate.
strings = ['abc', 'zeus', 'abstain', 'bow', 'zebra']
# strings = ['abc', 'zeus', 'abstain', 'bow', 'zebra']
for s in strings:
    root.add(s)
print (root)