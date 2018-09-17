
from collections import deque
from bisect import bisect_left, insort
from blist import blist


class RunningMedian:
    def __init__(self, iterable):
        self.q = q = deque(iterable)
        self.l = l = blist(q)
        l.sort()
        if len(q) % 2 == 1:
            self.odd = True
            self.mididx = (len(q) - 1) // 2
        else:
            self.odd = False
            self.mididx1 = len(q) // 2
            self.mididx2 = len(q) // 2 - 1

    def getMedian(self):
        l = self.l
        if self.odd:
            return l[self.mididx]
        else:
            return float(l[self.mididx1] + l[self.mididx2]) / 2

    def update(self, new_elem, check_invariants=False):
        q, l = self.q, self.l
        old_elem = q.popleft()
        q.append(new_elem)
        del l[bisect_left(l, old_elem)]
        insort(l, new_elem)

        if check_invariants:
            assert l == sorted(q)
        if self.odd:
            return l[self.mididx]
        else:
            return float(l[self.mididx1] + l[self.mididx2]) / 2


if __name__ == '__main__':
    from random import randrange
    from itertools import islice

    window_size = 6
    # data = [randrange(200) for i in range(1000)]
    data = [1, 2, 3, 4, 5, 6]

    it = iter(data)
    r = RunningMedian(islice(it, window_size))
    print(r.getMedian())
    print(r.update(2))
    print(r.update(2))
    print(r.update(2))
    print(r.update(2))
