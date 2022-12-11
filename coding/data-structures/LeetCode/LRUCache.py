class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.prev = None
        
class DLList:
    def __init__(self):
        self.head = Node('Dummy')
        self.last = self.head
        self.count = 0
       
    # def moveFront(self, n):
    #     # If node is already at front, nothing to do
    #     if self.head.next == n:
    #         return
    #     # if node is the last element, need to update last
    #     if self.last == n:
    #         self.last = self.last.prev
    #     if not (n.next and n.prev):
    #         self.count += 1
    #     # update relevant pointers for prev and next nodes
    #     if n.prev:
    #         n.prev.next = n.next
    #     if n.next:
    #         n.next.prev = n.prev
    #     # set node's prev and next pointers to place it in front
    #     n.next = self.head.next
    #     n.prev = self.head
    #     # update the new prev and next node's relevant pointers
    #     if self.head.next:
    #         self.head.next.prev = n
    #     else:
    #         self.last = n
    #     self.head.next = n
        
    def removeLast(self):
        bkp = self.last
        self.last = self.last.prev
        self.last.next = None
        self.count -= 1
        return bkp.val
        
    def insertFront(self, n, isNew=True):
        if isNew:
            self.count += 1
        n.next = self.head.next
        n.prev = self.head
        if self.head.next:
            self.head.next.prev = n
        else:
            self.last = n
        self.head.next = n
        
    def moveFront(self, n):
        if self.head.next == n:
            return
        if self.last == n:
            self.last = self.last.prev
        n.prev.next = n.next
        if n.next:
            n.next.prev = n.prev
        self.insertFront(n, False)
        
class LRUCache:

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise Exception("Invalid capacity")
        self.l = DLList()
        self.m = {}
        self.capacity = capacity
        

    def get(self, key: int) -> int:
        if key not in self.m:
            return -1
        n = self.m[key]
        self.l.moveFront(n)
        return n.val[1]
        

    def put(self, key: int, value: int) -> None:
        if key not in self.m:
            if self.l.count == self.capacity:
                k, v = self.l.removeLast()
                del(self.m[k])
            n = Node((key, value))
            self.m[key] = n
            self.l.insertFront(n)
        else:
            n = self.m[key]
            n.val = (key, value)
            self.l.moveFront(n)
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

c = LRUCache(2)
c.put(1,1)
c.put(2,2)
print (c.get(1))
c.put(3, 3)
print (c.get(2))
c.put(4,4)
print (c.get(1))