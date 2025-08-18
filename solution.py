import sys
input = sys.stdin.readline

MAXN = 100010
trans = [
    [2, 0, 0],   # state0: / -> 2, * -> 0, . -> 0
    [1, 3, 1],   # state1: / -> 1, * -> 3, . -> 1
    [2, 1, 0],   # state2: / -> 2, * -> 1, . -> 0
    [0, 3, 1]    # state3: / -> 0, * -> 3, . -> 1
]

class Node:
    def __init__(self):
        self.f = [i for i in range(4)]

def unit_node():
    return Node()

def make_node(c):
    node = Node()
    if c == '/':
        idx_char = 0
    elif c == '*':
        idx_char = 1
    else:
        idx_char = 2
    for s in range(4):
        node.f[s] = trans[s][idx_char]
    return node

def merge_node(left, right):
    res = Node()
    for s in range(4):
        res.f[s] = right.f[left.f[s]]
    return res

class SegmentTree:
    def __init__(self, s):
        self.n = len(s)
        self.tree = [Node() for _ in range(4 * self.n)]
        self.s_arr = list(s)
        if self.n > 0:
            self.build(0, 0, self.n - 1)

    def build(self, v, tl, tr):
        if tl == tr:
            self.tree[v] = make_node(self.s_arr[tl])
        else:
            tm = (tl + tr) // 2
            self.build(2*v+1, tl, tm)
            self.build(2*v+2, tm+1, tr)
            self.tree[v] = merge_node(self.tree[2*v+1], self.tree[2*v+2])

    def update(self, v, tl, tr, pos, c):
        if tl == tr:
            self.s_arr[tl] = c
            self.tree[v] = make_node(c)
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                self.update(2*v+1, tl, tm, pos, c)
            else:
                self.update(2*v+2, tm+1, tr, pos, c)
            self.tree[v] = merge_node(self.tree[2*v+1], self.tree[2*v+2])

    def query(self, v, tl, tr, l, r):
        if l > r:
            return unit_node()
        if l == tl and r == tr:
            return self.tree[v]
        tm = (tl + tr) // 2
        if r <= tm:
            return self.query(2*v+1, tl, tm, l, r)
        elif l > tm:
            return self.query(2*v+2, tm+1, tr, l, r)
        else:
            left_node = self.query(2*v+1, tl, tm, l, tm)
            right_node = self.query(2*v+2, tm+1, tr, tm+1, r)
            return merge_node(left_node, right_node)

def main():
    try:
        # Read input string
        s = input().strip()
        if not s:
            return
        
        # Initialize segment tree
        st = SegmentTree(s)
        n = len(s)
        
        # Read number of queries
        q = int(input())
        
        # Process queries
        for _ in range(q):
            parts = input().split()
            if not parts:
                continue
                
            op_type = int(parts[0])
            if op_type == 1:
                if len(parts) < 3:
                    continue
                pos = int(parts[1]) - 1  # Convert to 0-indexed
                c = parts[2]
                if 0 <= pos < n:  # Bounds check
                    st.update(0, 0, n - 1, pos, c)
            else:
                if len(parts) < 2:
                    continue
                pos = int(parts[1]) - 1  # Convert to 0-indexed
                if pos < 0:  # Bounds check
                    print("NO")
                    continue
                    
                if pos == 0:
                    state_before = 0
                else:
                    node = st.query(0, 0, n - 1, 0, pos - 1)
                    state_before = node.f[0]
                    
                print("YES" if state_before >= 1 else "NO")
                
    except Exception as e:
        # Handle any unexpected errors
        return

if __name__ == "__main__":
    main()