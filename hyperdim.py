import numpy as np
import pandas as pd
from typing import Any, Dict, List, DefaultDict, Optional
from collections import defaultdict
import scipy.special
from hashlib import md5
import random

# TODO:
# ClassicalHopfield


def spinor_to_binary(v: np.array) -> np.array:
    return (v + 1) / 2


def binary_to_spinor(v: np.array) -> np.array:
    return (v - 1 / 2) * 2


def array_to_int(v: np.array) -> int:
    r = 0
    for idx, x in enumerate(v):
        if x:
            r |= 1 << idx
    return r


def int_to_array(i: int, width: int) -> np.array:
    v = []
    for w in range(width):
        if i & 2 ** w:
            v.append(1)
        else:
            v.append(0)
    return np.array(v[::-1], dtype=bool)


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
        d = len(v)
        m = self.patterns * v
        arg = np.einsum("ij,jk->ik", m, (np.ones((d, d)) - np.diag(np.ones(d))))
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


def concat_ints(s: pd.Series) -> str:
    res = 0
    for x in s:
        res = (res << x.bit_length()) | x
    return str(res)


class ColumnHasher:
    def __init__(self, width: int):
        self.width = width
        self._over = (self.width // 64) + 1

    def hash(self, s: pd.Series) -> pd.Series:
        res = []
        for n in range(self._over):
            hash_key = md5(bytes(str(n), "utf8")).hexdigest()[:16]
            res.append(pd.util.hash_pandas_object(s, index=False, hash_key=hash_key))
        h = pd.concat(res, axis=1).apply(concat_ints, axis=1)
        return h.apply(lambda x: int(x) & int("1" * self.width, 2))


class ColumnBinder:
    def __init__(self, width: int, prng: np.random.RandomState, name=""):
        self.width = width
        self.hasher = ColumnHasher(width)
        # numpy truncates at 64 bits, need to stay in python
        self.map_key = random.randint(0, 2 ** width)

    def _map_hashes(self, s: pd.Series) -> pd.Series:
        return s.apply(lambda x: int(x) ^ self.map_key)

    def embed(self, s: pd.Series) -> pd.Series:
        hashes = self.hasher.hash(s)
        return self._map_hashes(hashes)

    def get_query_key(self, val: Any) -> int:
        return int(self.embed(pd.Series([val]))[0])


class RecordBinder:
    def __init__(self, columns: List[str], width: int, prng: np.random.RandomState):
        self.width = width
        self.binders = {x: ColumnBinder(width, prng, name=x) for x in columns}
        self._default_query_threshold = self._get_default_query_threshold(self.width)

    @staticmethod
    def _get_default_query_threshold(width, error_rate: float = 0.05) -> float:
        # TODO: this is wrong
        probs = np.array([scipy.special.binom(width, x) for x in range(width + 1)])
        probs_cdf = np.cumsum(probs) / np.sum(probs)
        return (width - np.max(np.where(probs_cdf < error_rate)[0])) / width

    # TODO: Add a method for aggregating rows of embeddings from records in a series
    def _agg_embeds(self, s: pd.Series, threshold=True) -> np.array:
        res = []
        for x in s.index:
            # s could have duplicate indicies
            if isinstance(s[x], int):
                res.append(int_to_array(s[x], self.width))
            else:
                for y in s[x]:
                    res.append(int_to_array(int(y), self.width))
        res = np.mean(np.array(res), axis=0)
        if threshold:
            res = res >= 0.5
        return array_to_int(res)

    def embed(self, df: pd.DataFrame) -> pd.Series:
        hashes = pd.concat([self.binders[x].embed(df[x]) for x in df.columns], axis=1)
        return hashes.apply(self._agg_embeds, axis=1)

    def get_query_key(self, query: Dict) -> np.array:
        tups = []
        for name, vals in query.items():
            if isinstance(vals, list):
                for v in vals:
                    tups.append((name, self.binders[name].get_query_key(v)))
            else:
                tups.append((name, self.binders[name].get_query_key(vals)))
        s = pd.DataFrame(tups).set_index(0)[1]
        return self._agg_embeds(s)

    def query(
        self, df: pd.DataFrame, query: Dict, threshold: float = None
    ) -> pd.DataFrame:
        threshold = (
            threshold if threshold is not None else self._default_query_threshold
        )

        key = self.get_query_key(query)
        embeds = self.embed(df)
        similarities = embeds.apply(
            lambda x: np.sum(
                ~np.logical_xor(
                    int_to_array(x, self.width), int_to_array(key, self.width)
                )
            )
            / self.width
        )
        similarities.name = "sim"
        return pd.merge(
            similarities.loc[similarities > threshold],
            df,
            left_index=True,
            right_index=True,
        )
