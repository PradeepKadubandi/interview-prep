# Tesla AP ML Infra — DS&A Solution Reference
 
All code below is verified (run against expected outputs). Python 3, standard library only.
Idioms to assume available: `collections` (`Counter`, `defaultdict`, `deque`), `heapq`, `bisect`. **Assume `sortedcontainers` is NOT in CoderPad.**
 
Format per problem: approach → complexity → code → gotchas.
 
---
 
## Arrays & Hashing
 
### Two Sum (1)
One pass, store value→index; check complement before inserting. **O(n) / O(n)**.
```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []
```
Gotcha: check complement *before* inserting `x`, or you may match an element with itself.
 
### Group Anagrams (49)
Key each word by its sorted tuple. **O(n·k log k)**.
```python
from collections import defaultdict
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        groups[tuple(sorted(s))].append(s)
    return list(groups.values())
```
Gotcha: dict keys must be hashable → use `tuple(sorted(s))`, not the list. A 26-length count tuple is O(n·k) if asked to optimize.
 
### Top K Frequent Elements (347)
Know both. Heap: **O(n log k)**. Bucket sort by frequency: **O(n)**.
```python
from collections import Counter
def top_k_frequent(nums, k):
    return [x for x, _ in Counter(nums).most_common(k)]
 
def top_k_frequent_bucket(nums, k):
    cnt = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]   # index = frequency
    for x, c in cnt.items():
        buckets[c].append(x)
    out = []
    for c in range(len(buckets) - 1, 0, -1):
        for x in buckets[c]:
            out.append(x)
            if len(out) == k:
                return out
    return out
```
Gotcha: interviewer often pushes from `most_common` → "can you beat O(n log n)?" Bucket sort is the answer (frequency is bounded by n).
 
### Product of Array Except Self (238)
Prefix products left-to-right, then multiply suffix products right-to-left in place. **O(n) / O(1)** extra (output aside). No division.
```python
def product_except_self(nums):
    n = len(nums)
    res = [1] * n
    pre = 1
    for i in range(n):
        res[i] = pre
        pre *= nums[i]
    suf = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suf
        suf *= nums[i]
    return res
```
Gotcha: the "no division" constraint is the whole point — don't reach for it even if zeros tempt you.
 
### Valid Anagram (242) / Contains Duplicate (217)
```python
def is_anagram(s, t): return Counter(s) == Counter(t)
def contains_duplicate(nums): return len(set(nums)) != len(nums)
```
 
### Longest Consecutive Sequence (128)
Set membership; only start counting at a run's left edge (`x-1 not in set`). **O(n)**.
```python
def longest_consecutive(nums):
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 not in s:            # left edge only → each run counted once
            length = 1
            while x + length in s:
                length += 1
            best = max(best, length)
    return best
```
Gotcha: the `x-1 not in s` guard is what keeps it O(n); without it the inner while makes it O(n²).
 
---
 
## Two Pointers / Sliding Window
 
### Longest Substring Without Repeating Characters (3)
Track last-seen index; jump `left` past the duplicate. **O(n)**.
```python
def length_of_longest_substring(s):
    last = {}
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = right
        best = max(best, right - left + 1)
    return best
```
Gotcha: the `last[ch] >= left` check matters — a stale duplicate *behind* `left` must not pull the window backward.
 
### 3Sum (15)
Sort, fix `i`, two-pointer the rest. Skip duplicates at all three positions. **O(n²)**.
```python
def three_sum(nums):
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        if nums[i] > 0:
            break
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:   lo += 1
            elif s > 0: hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                lo += 1; hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]: lo += 1
                while lo < hi and nums[hi] == nums[hi + 1]: hi -= 1
    return res
```
Gotcha: dedup logic is where people lose points — skip dup `i`, and skip dup `lo`/`hi` *after* recording a hit.
 
### Maximum Subarray (53) — Kadane
```python
def max_subarray(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)        # extend or restart
        best = max(best, cur)
    return best
```
Gotcha: initialize with `nums[0]`, not 0 — handles all-negative arrays.
 
### Container With Most Water (11)
Move the shorter wall inward. **O(n)**.
```python
def max_area(height):
    lo, hi, best = 0, len(height) - 1, 0
    while lo < hi:
        best = max(best, (hi - lo) * min(height[lo], height[hi]))
        if height[lo] < height[hi]: lo += 1
        else: hi -= 1
    return best
```
 
### Valid Palindrome (125)
Two pointers, skip non-alphanumeric, case-insensitive.
```python
def is_palindrome(s):
    lo, hi = 0, len(s) - 1
    while lo < hi:
        while lo < hi and not s[lo].isalnum(): lo += 1
        while lo < hi and not s[hi].isalnum(): hi -= 1
        if s[lo].lower() != s[hi].lower(): return False
        lo += 1; hi -= 1
    return True
```
 
### Longest Repeating Character Replacement (424)
Window valid while `(window_len − max_freq) <= k`. **O(n)**.
```python
from collections import defaultdict
def character_replacement(s, k):
    cnt = defaultdict(int)
    left = best = max_freq = 0
    for right, ch in enumerate(s):
        cnt[ch] += 1
        max_freq = max(max_freq, cnt[ch])
        while (right - left + 1) - max_freq > k:
            cnt[s[left]] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```
Gotcha: `max_freq` is intentionally not decreased when shrinking — it can stay "stale high," but that only ever lets the window grow, which is correct and keeps it O(n).
 
### Minimum Window Substring (76) — *hard*
Expand right, contract left while valid; track best. **O(|s| + |t|)**.
```python
from collections import Counter
def min_window(s, t):
    if not t or not s: return ""
    need = Counter(t)
    missing = len(t)
    left = start = 0
    end = float("inf")
    for right, ch in enumerate(s):
        if need[ch] > 0: missing -= 1
        need[ch] -= 1
        while missing == 0:                 # window covers t → try to shrink
            if right - left < end - start:
                start, end = left, right
            need[s[left]] += 1
            if need[s[left]] > 0: missing += 1
            left += 1
    return "" if end == float("inf") else s[start:end + 1]
```
Gotcha: `need` goes negative for over-represented chars — that's how you know shrinking past them is safe.
 
### Trapping Rain Water (42) — *hard*
Two pointers from both ends, advance the side with the smaller running max. **O(n) / O(1)**.
```python
def trap(height):
    lo, hi = 0, len(height) - 1
    left_max = right_max = water = 0
    while lo < hi:
        if height[lo] < height[hi]:
            left_max = max(left_max, height[lo])
            water += left_max - height[lo]; lo += 1
        else:
            right_max = max(right_max, height[hi])
            water += right_max - height[hi]; hi -= 1
    return water
```
Gotcha: the side with the smaller bar is the bottleneck, so it's always safe to settle that side's water.
 
---
 
## Stack
 
### Valid Parentheses (20)
```python
def valid_parentheses(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    st = []
    for ch in s:
        if ch in pairs:
            if not st or st.pop() != pairs[ch]: return False
        else:
            st.append(ch)
    return not st
```
Gotcha: return `not st` at the end — unmatched openers must fail.
 
### Min Stack (155)
Store `(value, running_min)` pairs → O(1) `getMin`.
```python
class MinStack:
    def __init__(self): self.st = []
    def push(self, val):
        m = val if not self.st else min(val, self.st[-1][1])
        self.st.append((val, m))
    def pop(self): self.st.pop()
    def top(self): return self.st[-1][0]
    def getMin(self): return self.st[-1][1]
```
 
### Daily Temperatures (739) — monotonic stack
Stack of indices with decreasing temps; pop when a warmer day arrives. **O(n)**.
```python
def daily_temperatures(temps):
    res = [0] * len(temps)
    st = []                              # indices, temps decreasing
    for i, t in enumerate(temps):
        while st and temps[st[-1]] < t:
            j = st.pop(); res[j] = i - j
        st.append(i)
    return res
```
 
### Largest Rectangle in Histogram (84) — *hard*
Monotonic increasing stack; on pop, width spans from the new left boundary. Sentinel `0` flushes the stack. **O(n)**.
```python
def largest_rectangle(heights):
    st = []                              # indices, heights increasing
    best = 0
    for i, h in enumerate(heights + [0]):
        while st and heights[st[-1]] >= h:
            height = heights[st.pop()]
            width = i if not st else i - st[-1] - 1
            best = max(best, height * width)
        st.append(i)
    return best
```
Gotcha: the trailing `[0]` sentinel guarantees every bar gets popped and measured.
 
---
 
## Binary Search
 
### Search in Rotated Sorted Array (33)
One half is always sorted; decide which, then whether target lies inside it. **O(log n)**.
```python
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target: return mid
        if nums[lo] <= nums[mid]:                    # left sorted
            if nums[lo] <= target < nums[mid]: hi = mid - 1
            else: lo = mid + 1
        else:                                        # right sorted
            if nums[mid] < target <= nums[hi]: lo = mid + 1
            else: hi = mid - 1
    return -1
```
Gotcha: `nums[lo] <= nums[mid]` (≤, not <) handles the 2-element / equal case.
 
### Koko Eating Bananas (875) — binary search on the answer
Search the *speed*; feasibility is monotonic. **O(n log max)**.
```python
import math
def min_eating_speed(piles, h):
    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(math.ceil(p / mid) for p in piles) <= h: hi = mid
        else: lo = mid + 1
    return lo
```
Gotcha: this "binary search on a monotonic predicate" pattern is very Tesla-flavored (practical optimization). Recognize the shape: minimize X subject to a monotonic feasibility test.
 
### Find Minimum in Rotated Sorted Array (153)
Compare mid to `hi` to decide which half holds the pivot. **O(log n)**.
```python
def find_min_rotated(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[hi]: lo = mid + 1
        else: hi = mid
    return nums[lo]
```
Gotcha: compare against `hi`, not `lo` — comparing to `lo` breaks on already-sorted input.
 
### Median of Two Sorted Arrays (4) — *hard*
Binary search the partition of the shorter array so left halves ≤ right halves. **O(log min(m,n))**.
```python
def find_median_sorted_arrays(a, b):
    if len(a) > len(b): a, b = b, a
    m, n = len(a), len(b)
    lo, hi, half = 0, m, (m + n + 1) // 2
    while lo <= hi:
        i = (lo + hi) // 2
        j = half - i
        aL = a[i-1] if i > 0 else float("-inf")
        aR = a[i]   if i < m else float("inf")
        bL = b[j-1] if j > 0 else float("-inf")
        bR = b[j]   if j < n else float("inf")
        if aL <= bR and bL <= aR:
            if (m + n) % 2: return max(aL, bL)
            return (max(aL, bL) + min(aR, bR)) / 2
        elif aL > bR: hi = i - 1
        else: lo = i + 1
```
Gotcha: the ±inf sentinels remove all the edge-case branching. Always binary-search the *shorter* array.
 
---
 
## Linked List
 
Helper node:
```python
class ListNode:
    def __init__(self, val=0, nxt=None):
        self.val = val; self.next = nxt
```
 
### Reverse Linked List (206)
```python
def reverse_list(head):
    prev = None
    while head:
        head.next, prev, head = prev, head, head.next
    return prev
```
Gotcha: the tuple-assignment order matters — RHS is evaluated first, so this is safe; if you unroll it, save `head.next` before overwriting.
 
### Merge Two Sorted Lists (21)
Dummy head + tail pointer; attach remainder. **O(m+n)**.
```python
def merge_two_lists(a, b):
    dummy = tail = ListNode()
    while a and b:
        if a.val <= b.val: tail.next, a = a, a.next
        else: tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next
```
 
### Linked List Cycle (141) — Floyd's
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast: return True
    return False
```
 
### LRU Cache (146) — *know cold*
Hashmap + doubly-linked list. Both `get` and `put` are **O(1)**. Most-recent at front, evict from back.
```python
class Node:
    def __init__(self, key=0, val=0):
        self.key, self.val = key, val
        self.prev = self.next = None
 
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}
        self.head, self.tail = Node(), Node()
        self.head.next, self.tail.prev = self.tail, self.head
    def _remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev
    def _add_front(self, node):
        node.prev, node.next = self.head, self.head.next
        self.head.next.prev = node
        self.head.next = node
    def get(self, key):
        if key not in self.cache: return -1
        node = self.cache[key]
        self._remove(node); self._add_front(node)
        return node.val
    def put(self, key, value):
        if key in self.cache: self._remove(self.cache[key])
        node = Node(key, value)
        self.cache[key] = node; self._add_front(node)
        if len(self.cache) > self.cap:
            lru = self.tail.prev
            self._remove(lru); del self.cache[lru.key]
```
Gotcha: store the `key` on the node so eviction can delete from the dict. Use dummy head+tail to kill all the null checks. (`functools.lru_cache` won't satisfy a "design it" prompt.)
 
### Merge k Sorted Lists (23)
Min-heap of (val, tiebreak, node). **O(N log k)**.
```python
import heapq
def merge_k_lists(lists):
    h = []
    for i, node in enumerate(lists):
        if node: heapq.heappush(h, (node.val, i, node))
    dummy = tail = ListNode()
    while h:
        val, i, node = heapq.heappop(h)
        tail.next = node; tail = node
        if node.next: heapq.heappush(h, (node.next.val, i, node.next))
    return dummy.next
```
Gotcha: include the list index `i` as a tiebreaker — heapq will otherwise try to compare `ListNode`s and crash on equal values.
 
---
 
## Trees
 
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right
```
 
### Binary Tree Level Order Traversal (102) — BFS
```python
from collections import deque
def level_order(root):
    if not root: return []
    res, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):              # snapshot this level's size
            n = q.popleft(); level.append(n.val)
            if n.left: q.append(n.left)
            if n.right: q.append(n.right)
        res.append(level)
    return res
```
Gotcha: `for _ in range(len(q))` fixes the level boundary before you start adding the next level.
 
### Validate BST (98)
Carry down (lo, hi) bounds — not just parent comparison. **O(n)**.
```python
def is_valid_bst(root):
    def dfs(node, lo, hi):
        if not node: return True
        if not (lo < node.val < hi): return False
        return dfs(node.left, lo, node.val) and dfs(node.right, node.val, hi)
    return dfs(root, float("-inf"), float("inf"))
```
Gotcha: comparing only against the immediate parent is the classic wrong answer — a deep descendant can violate an ancestor's bound.
 
### Lowest Common Ancestor (236)
Bubble up: node is LCA if `p` and `q` split across its subtrees.
```python
def lca(root, p, q):
    if not root or root.val == p or root.val == q: return root
    L, R = lca(root.left, p, q), lca(root.right, p, q)
    if L and R: return root
    return L or R
```
 
### Kth Smallest in BST (230)
Iterative in-order; in-order of a BST is sorted. **O(h + k)**.
```python
def kth_smallest(root, k):
    st, cur = [], root
    while st or cur:
        while cur: st.append(cur); cur = cur.left
        cur = st.pop(); k -= 1
        if k == 0: return cur.val
        cur = cur.right
```
 
### Invert (226) / Max Depth (104)
```python
def invert_tree(root):
    if root:
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
def max_depth(root):
    return 1 + max(max_depth(root.left), max_depth(root.right)) if root else 0
```
 
### Serialize / Deserialize Binary Tree (297) — *hard*
Pre-order with `#` for null; reconstruct from an iterator. **O(n)**.
```python
class Codec:
    def serialize(self, root):
        out = []
        def dfs(n):
            if not n: out.append("#"); return
            out.append(str(n.val)); dfs(n.left); dfs(n.right)
        dfs(root); return ",".join(out)
    def deserialize(self, data):
        vals = iter(data.split(","))
        def dfs():
            v = next(vals)
            if v == "#": return None
            n = TreeNode(int(v)); n.left = dfs(); n.right = dfs(); return n
        return dfs()
```
Gotcha: an iterator (not an index) keeps the recursion clean — pre-order serialize pairs with pre-order rebuild.
 
---
 
## Heaps
 
### Kth Largest Element (215)
Min-heap of size k. **O(n log k)**. (Quickselect gives avg O(n) — mention it.)
```python
import heapq
def kth_largest(nums, k):
    h = nums[:k]; heapq.heapify(h)
    for x in nums[k:]:
        if x > h[0]: heapq.heapreplace(h, x)
    return h[0]
```
Gotcha: keep a *min*-heap of the k largest (the root is the kth largest). Don't push all n then pop k — that's O(n log n).
 
### Find Median from Data Stream (295) — *hard*, two heaps
Max-heap (lower half) + min-heap (upper half), kept balanced. add **O(log n)**, find **O(1)**.
```python
import heapq
class MedianFinder:
    def __init__(self):
        self.lo = []   # max-heap via negation
        self.hi = []   # min-heap
    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))   # funnel max(lo) into hi
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))
    def findMedian(self):
        if len(self.lo) > len(self.hi): return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```
Gotcha: Python only has a min-heap → negate for the max-heap. The push-to-lo-then-shift-to-hi trick keeps both halves correctly ordered.
 
### Task Scheduler (621)
Greedy/math: most-frequent task fixes the schedule frame. **O(n)**.
```python
from collections import Counter
def task_scheduler(tasks, n):
    cnt = Counter(tasks)
    f_max = max(cnt.values())
    n_max = sum(1 for v in cnt.values() if v == f_max)
    return max(len(tasks), (f_max - 1) * (n + 1) + n_max)
```
Gotcha: `max(len(tasks), …)` covers the case where there are enough distinct tasks that no idle is needed.
 
---
 
## Graphs
 
### Number of Islands (200) — DFS flood fill
```python
def num_islands(grid):
    if not grid: return 0
    rows, cols = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] != "1": return
        grid[r][c] = "0"
        dfs(r+1,c); dfs(r-1,c); dfs(r,c+1); dfs(r,c-1)
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1; dfs(r, c)
    return count
```
Gotcha: mark visited by mutating to `"0"` (or a visited set if you can't mutate input — ask). Deep grids can blow the recursion limit → mention an explicit stack / BFS.
 
### Course Schedule (207) — topological sort (Kahn's)
Cycle detection: can all nodes reach in-degree 0? **O(V+E)**.
```python
from collections import defaultdict, deque
def can_finish(num_courses, prerequisites):
    graph = defaultdict(list); indeg = [0] * num_courses
    for a, b in prerequisites:
        graph[b].append(a); indeg[a] += 1
    q = deque(i for i in range(num_courses) if indeg[i] == 0)
    seen = 0
    while q:
        node = q.popleft(); seen += 1
        for nxt in graph[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0: q.append(nxt)
    return seen == num_courses
```
Gotcha: if `seen < num_courses`, a cycle exists. Get the edge direction right (`b → a`: b is prereq of a).
 
### Rotting Oranges (994) — multi-source BFS
Seed the queue with *all* rotten cells at t=0. **O(rows·cols)**.
```python
from collections import deque
def oranges_rotting(grid):
    rows, cols = len(grid), len(grid[0])
    q = deque(); fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2: q.append((r, c, 0))
            elif grid[r][c] == 1: fresh += 1
    minutes = 0
    while q:
        r, c, t = q.popleft(); minutes = max(minutes, t)
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2; fresh -= 1; q.append((nr, nc, t+1))
    return minutes if fresh == 0 else -1
```
Gotcha: track `fresh` to detect unreachable oranges → return -1. Multi-source seeding is the key idea.
 
### Clone Graph (133)
DFS/BFS with a `old→new` map to handle cycles.
```python
def clone_graph(node):
    if not node: return None
    clones = {}
    def dfs(n):
        if n in clones: return clones[n]
        copy = Node(n.val)              # assumes a Node class with .val/.neighbors
        clones[n] = copy
        copy.neighbors = [dfs(nb) for nb in n.neighbors]
        return copy
    return dfs(node)
```
Gotcha: insert into the map *before* recursing into neighbors, or cycles recurse forever.
 
### Network Delay Time (743) — Dijkstra
Lazy Dijkstra with a min-heap. **O(E log V)**.
```python
import heapq
from collections import defaultdict
def network_delay_time(times, n, k):
    graph = defaultdict(list)
    for u, v, w in times: graph[u].append((v, w))
    dist = {}
    h = [(0, k)]
    while h:
        d, node = heapq.heappop(h)
        if node in dist: continue           # already finalized
        dist[node] = d
        for nxt, w in graph[node]:
            if nxt not in dist:
                heapq.heappush(h, (d + w, nxt))
    return max(dist.values()) if len(dist) == n else -1
```
Gotcha: the `if node in dist: continue` is the lazy-deletion guard — without it you reprocess stale heap entries. This is the template for Tesla's charging-station shortest-path style problem.
 
Finalize on **pop**, not on push. A node can sit in the heap multiple times with different distances (e.g. a long edge pushed first, a shorter multi-hop path pushed later); the min-heap pops the *smallest* first, so the first pop is always the shortest and later pops are stale (≥) and skipped. Marking a node finalized at *push* time instead would lock in the first-discovered (not shortest) distance and silently drop the better path — the classic wrong-Dijkstra. The first-pop-is-shortest invariant relies on **non-negative weights**; with negative edges it breaks and you need Bellman-Ford/SPFA (not a concern for 743, where `1 ≤ w ≤ 100`).
 
---
 
## Intervals
 
### Merge Intervals (56)
Sort by start; extend or append. **O(n log n)**.
```python
def merge_intervals(intervals):
    intervals.sort()
    res = []
    for s, e in intervals:
        if res and s <= res[-1][1]:
            res[-1][1] = max(res[-1][1], e)
        else:
            res.append([s, e])
    return res
```
 
### Insert Interval (57)
Three phases: before, overlapping (merge), after. **O(n)**.
```python
def insert_interval(intervals, new):
    res = []; i, n = 0, len(intervals)
    while i < n and intervals[i][1] < new[0]:
        res.append(intervals[i]); i += 1
    while i < n and intervals[i][0] <= new[1]:
        new = [min(new[0], intervals[i][0]), max(new[1], intervals[i][1])]; i += 1
    res.append(new)
    while i < n:
        res.append(intervals[i]); i += 1
    return res
```
 
### Non-overlapping Intervals (435) — greedy
Sort by **end**; keep the earliest-ending non-conflicting interval. **O(n log n)**.
```python
def erase_overlap_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count, end = 0, float("-inf")
    for s, e in intervals:
        if s >= end: end = e
        else: count += 1                    # drop this one
    return count
```
Gotcha: sort by end, not start — that's the classic activity-selection greedy.
 
### Meeting Rooms II (253)
Min-heap of end times = rooms in use. **O(n log n)**.
```python
import heapq
def min_meeting_rooms(intervals):
    intervals.sort()
    h = []
    for s, e in intervals:
        if h and h[0] <= s: heapq.heapreplace(h, e)   # reuse a freed room
        else: heapq.heappush(h, e)
    return len(h)
```
 
---
 
## DP & Greedy
 
### Coin Change (322)
Bottom-up over amounts. **O(amount·coins)**.
```python
def coin_change(coins, amount):
    dp = [0] + [float("inf")] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a: dp[a] = min(dp[a], dp[a-c] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1
```
 
### Word Break (139)
`dp[i]` = prefix of length i is segmentable. **O(n²)** (with set lookups).
```python
def word_break(s, words):
    wset = set(words); n = len(s)
    dp = [False] * (n + 1); dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in wset:
                dp[i] = True; break
    return dp[n]
```
 
### Longest Increasing Subsequence (300)
Patience sorting with `bisect`. **O(n log n)**.
```python
import bisect
def length_of_lis(nums):
    sub = []
    for x in nums:
        i = bisect.bisect_left(sub, x)
        if i == len(sub): sub.append(x)
        else: sub[i] = x
    return len(sub)
```
Gotcha: `sub` is NOT a real subsequence — only its *length* is meaningful. `bisect_left` (not right) for strictly increasing.
 
### Climbing Stairs (70) / House Robber (198)
```python
def climb_stairs(n):
    a, b = 1, 1
    for _ in range(n): a, b = b, a + b
    return a
def rob(nums):
    prev = cur = 0
    for x in nums:
        prev, cur = cur, max(cur, prev + x)
    return cur
```
 
### Unique Paths (62)
Rolling 1-D DP. **O(m·n) / O(n)**.
```python
def unique_paths(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n): dp[j] += dp[j-1]
    return dp[-1]
```
 
### Longest Common Subsequence (1143) / Edit Distance (72)
```python
def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
 
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```
Gotcha (edit distance): seed the first row/column with 0..n (cost of pure inserts/deletes), then the three-way min covers insert/delete/replace.
 
### Jump Game (55) / Gas Station (134) — greedy
```python
def can_jump(nums):
    reach = 0
    for i, x in enumerate(nums):
        if i > reach: return False
        reach = max(reach, i + x)
    return True
 
def gas_station(gas, cost):
    if sum(gas) < sum(cost): return -1      # feasibility check
    start = tank = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0: start = i + 1; tank = 0  # can't reach i+1 from current start
    return start
```
Gotcha (gas): if total gas ≥ total cost a solution is guaranteed unique; the running-deficit reset finds it in one pass.
 
---
 
## Matrix
 
### Rotate Image (48)
Reverse rows, then transpose → 90° clockwise, in place.
```python
def rotate_image(matrix):
    matrix.reverse()
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix
```
 
### Spiral Matrix (54)
Peel the top row, rotate the rest counter-clockwise, repeat.
```python
def spiral_order(matrix):
    res = []
    while matrix:
        res += matrix.pop(0)
        matrix = [list(row) for row in zip(*matrix)][::-1]
    return res
```
 
### Set Matrix Zeroes (73)
Use first row/col as marker storage → **O(1)** extra space.
```python
def set_zeroes(matrix):
    rows, cols = len(matrix), len(matrix[0])
    first_row = any(matrix[0][c] == 0 for c in range(cols))
    first_col = any(matrix[r][0] == 0 for r in range(rows))
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][c] == 0:
                matrix[r][0] = matrix[0][c] = 0
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0
    if first_row:
        for c in range(cols): matrix[0][c] = 0
    if first_col:
        for r in range(rows): matrix[r][0] = 0
    return matrix
```
Gotcha: handle the first row/col flags *separately and last*, or the markers overwrite themselves.
 
---
 
## Quick reference — Python idioms that save time
 
- `Counter(x).most_common(k)` → top-k by frequency.
- `heapq` is a **min**-heap; negate values for a max-heap.
- `heapq.heapreplace(h, x)` = pop-then-push in one O(log n) op; `heappushpop` pushes first.
- `bisect.bisect_left / insort` for sorted-array ops (LIS, running medians without a heap).
- `deque` for O(1) BFS pops; `popleft()`.
- Tuples in a heap need a tiebreaker before any non-comparable object (e.g. nodes).
- `float("inf") / float("-inf")` as DP/bounds sentinels to delete edge-case branches.