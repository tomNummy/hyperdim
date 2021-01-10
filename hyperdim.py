import numpy as np
from typing import Dict, List, DefaultDict, Optional
from collections import defaultdict

# TODO:
# ClassicalHopfield


def spinor_to_binary(v: np.array) -> np.array:
    return (v + 1) / 2


def binary_to_spinor(v: np.array) -> np.array:
    return (v - 1 / 2) * 2


def arr_to_binary(v: np.array) -> int:
    return int(v.dot(1 << np.arange(v.shape[-1] - 1, -1, -1)))


def hamming_dist(x: int, y: int) -> int:
    return bin(x ^ y).count("1")


class RSP:
    _plane: np.array

    def __init__(self, n, seed: int = None) -> None:
        prng = np.random.RandomState(seed)
        self._plane = prng.choice([-1, 1], size=n)

    def hash(self, v: np.array) -> int:
        return np.sign(np.dot(v, self._plane))


class RSPHash:
    _hashers: List[RSP]

    def __init__(
        self, dim: int = None, width: int = None, rsps: List[RSP] = None
    ) -> None:
        if rsps:
            self._hashers = np.array(rsps)
        elif dim and width:
            self._hashers = np.array([RSP(dim) for _ in range(width)])
        else:
            raise ValueError()

    def hash(self, v: np.array) -> int:
        res = np.vectorize(lambda x: x.hash(v))(self._hashers)
        return arr_to_binary(spinor_to_binary(res))


class ClassicHopfield:
    MAX_UPDATES = 2
    W: np.array

    def __init__(self, dim: int):
        self.W = np.zeros((dim, dim))

    def learn(self, v: np.array) -> None:
        m = np.einsum("i,j->ij", v, v)
        m -= np.diag(m.diagonal())
        self.W += m

    def recall(self, v: np.array) -> Optional[np.array]:
        v_old = v
        for n in range(self.MAX_UPDATES):
            v_new = self.update(v_old)
            diff = np.diff(np.vstack([v_old, v_new]), axis=0)
            if np.all(diff == 0):
                return v_new
            v_old = v_new
        print(f"Failed to converge ... {np.sum(diff != 0)} components")
        return None

    def update(self, v: np.array) -> np.array:
        return np.sign(np.einsum("ij,i->j", self.W, v))


class ModernHopfield:
    MAX_UPDATES = 2
    patterns: np.array

    def __init__(self, dim: int):
        self.patterns = None

    def learn(self, v: np.array) -> None:
        if self.patterns is None:
            self.patterns = v[None, :]
        else:
            self.patterns = np.concatenate([self.patterns, v[None, :]])

    def update_kernel(self, x: np.array) -> np.array:
        return np.exp(x)

    def update(self, v) -> np.array:
        arg = self.patterns * v - np.dot(self.patterns, v)
        pos = arg + self.patterns
        neg = arg - self.patterns
        return np.sign(
            np.sum(self.update_kernel(pos) - self.update_kernel(neg), axis=0)
        )

    def recall(self, v: np.array) -> Optional[np.array]:
        v_old = v
        for n in range(self.MAX_UPDATES):
            v_new = self.update(v_old)
            diff = np.diff(np.vstack([v_old, v_new]), axis=0)
            if np.all(diff == 0):
                return v_new
            v_old = v_new
        print(f"Failed to converge ... {np.sum(diff != 0)} components")
        return None


class RSPMem:
    """
    Use RSP LSH to dynamically build hopfield networks with only
    nearest neighbors.
    Can be used as an efficient associative memory.
    """

    hasher: RSPHash
    patterns: DefaultDict[int, List[np.array]]
    dim: int

    def __init__(
        self,
        dim: int,
        width: int,
    ):
        self.dim = dim
        self.hasher = RSPHash(dim, width)
        self.patterns = defaultdict(list)

    def learn(self, v: np.array):
        self.patterns[self.hasher.hash(v)].append(v)

    def recall(self, v: np.array):
        # find the buckets
        h = self.hasher.hash(v)
        buckets = {x for x in self.patterns if hamming_dist(h, x) <= 1}
        patterns = []
        for b in buckets:
            patterns.extend(self.patterns[b])
        print(f"NN patterns: {len(patterns)}")
        assert patterns
        mem = self._build_memory(patterns)
        return mem.recall(v)

    def _build_memory(self, patterns: List[np.array]) -> ModernHopfield:
        hf = ModernHopfield(self.dim)
        for pattern in patterns:
            hf.learn(pattern)
        return hf
