from collections import deque, OrderedDict, Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def gen_trace(n=20000, pages=200, seed=0, ws=12, phase="both"):
    rng = np.random.default_rng(seed)
    half = n // 2

    def p1(m):
        out, base, seq = [], int(rng.integers(0, pages)), int(rng.integers(0, pages))
        w = 1.0 / np.arange(1, ws + 1)
        w /= w.sum()
        for i in range(m):
            if i % 500 == 0 and i:
                base = (base + 4) % pages
            if rng.random() < 0.9:
                out.append((base + int(rng.choice(ws, p=w))) % pages)
            else:
                seq = (seq + 1) % pages
                out.append(seq)
        return out

    def p2(m):
        out = []
        while len(out) < m:
            if rng.random() < 0.05:
                cl = rng.integers(0, pages, size=3)
                for _ in range(int(rng.integers(5, 21))):
                    out.append(int(rng.choice(cl)))
            else:
                out.append(int(rng.integers(0, pages)))
        return out[:m]

    if phase == "p1":
        return np.array(p1(n)), n
    if phase == "p2":
        return np.array(p2(n)), 0
    return np.array(p1(half) + p2(n - half)), half


def fifo(trace, frames):
    q, s = deque(), set()
    hits = np.zeros(len(trace), bool)
    for i, p in enumerate(trace):
        if p in s:
            hits[i] = True
            continue
        if len(q) == frames:
            s.discard(q.popleft())
        q.append(p)
        s.add(p)
    return hits


def lru(trace, frames):
    od = OrderedDict()
    hits = np.zeros(len(trace), bool)
    for i, p in enumerate(trace):
        if p in od:
            hits[i] = True
            od.move_to_end(p)
            continue
        if len(od) == frames:
            od.popitem(last=False)
        od[p] = True
    return hits


def next_occurrence(trace):
    n = len(trace)
    nxt, last = np.full(n, n, dtype=np.int64), {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(trace[i], n)
        last[trace[i]] = i
    return nxt


def opt(trace, frames):
    nxt = next_occurrence(trace)
    nu, hits = {}, np.zeros(len(trace), bool)
    for i, p in enumerate(trace):
        if p in nu:
            hits[i] = True
        elif len(nu) == frames:
            victim = max(nu, key=nu.get)
            del nu[victim]
        nu[p] = nxt[i]
    return hits


W = 100


class State:

    def __init__(self):
        self.last, self.load, self.total = {}, {}, Counter()
        self.win, self.wcount = deque(), Counter()

    def touch(self, t, p, loaded):
        if loaded:
            self.load[p] = t
            self.total[p] = 0
        self.last[p] = t
        self.total[p] += 1
        self.win.append(p)
        self.wcount[p] += 1
        if len(self.win) > W:
            self.wcount[self.win.popleft()] -= 1

    def evict(self, p):
        for d in (self.last, self.load, self.total):
            d.pop(p, None)

    def feats(self, t, pages):
        return np.array([[t - self.last[p], self.wcount[p], t - self.load[p], self.total[p]]
                         for p in pages], dtype=float)


def collect(trace, frames, D, behaviour="lru", stop=None):
    nxt = next_occurrence(trace)
    stop = len(trace) if stop is None else stop
    od, st, last_i = OrderedDict(), State(), {}
    X, y = [], []
    for t in range(stop):
        p = trace[t]
        if p in od:
            od.move_to_end(p)
            st.touch(t, p, False)
            last_i[p] = t
            continue
        if len(od) == frames:
            res = list(od.keys())
            X.append(st.feats(t, res))
            y.extend(int(nxt[last_i[q]] - t > D) for q in res)
            v = next(iter(od))
            del od[v]
            st.evict(v)
        od[p] = True
        st.touch(t, p, True)
        last_i[p] = t
    if not X:
        return np.empty((0, 4)), np.empty(0)
    return np.vstack(X), np.array(y)


def make_model(kind="tree"):
    if kind == "logreg":
        return LogisticRegression(max_iter=500, class_weight="balanced")
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, class_weight="balanced",
                                  random_state=0)


def fit(X, y, kind="tree"):
    m = make_model(kind)
    if len(np.unique(y)) < 2:
        return None
    return m.fit(X, y)


def learned(trace, frames, model, D, online=False, retrain_every=1000, window=3000, kind="tree"):
    od, st = OrderedDict(), State()
    hits = np.zeros(len(trace), bool)
    for t, p in enumerate(trace):
        if online and t and t % retrain_every == 0:
            lo = max(0, t - window)
            sub = trace[lo:t]
            X, y = collect(sub, frames, D, stop=max(0, len(sub) - D))
            if len(y) > 50:
                m = fit(X, y, kind)
                if m is not None:
                    model = m
        if p in od:
            hits[t] = True
            od.move_to_end(p)
            st.touch(t, p, False)
            continue
        if len(od) == frames:
            res = list(od.keys())
            if model is None:
                v = res[0]
            else:
                score = model.predict_proba(st.feats(t, res))[:, list(model.classes_).index(1)]
                score = score + 1e-9 * np.arange(len(res))[::-1]
                v = res[int(np.argmax(score))]
            del od[v]
            st.evict(v)
        od[p] = True
        st.touch(t, p, True)
    return hits
