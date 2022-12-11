class Point:
    def __init__(self, (x,y)):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        if isinstance(other, Point):
            return Point((self.x + other.x, self.y + other.y))
        return None

    def __iadd__(self, other):
        if isinstance(other, Point):
            self.x += other.x
            self.y += other.y
            return self
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point((self.x - other.x, self.y - other.y))
        return None

    def __isub__(self, other):
        if isinstance(other, Point):
            self.x -= other.x
            self.y -= other.y
            return self
        return NotImplemented

if __name__ == '__main__':
    # Crude testing
    t = Point((2,3))
    assert t.x == 2 and t.y == 3, 'Construction Unexpected Result'
    t += Point((1,2))
    assert t.x == 3 and t.y == 5, 'iAdd Unexpected Result'
    n_t = t + Point((4, 3))
    assert n_t.x == 7 and n_t.y == 8, 'Add Unexpected Result'
    n_t -= t
    assert n_t.x == 4 and n_t.y == 3, 'iSub Unexpected Result'
    n_t = n_t - n_t
    assert n_t.x == 0 and n_t.y == 0, 'Sub Unexpected Result'
