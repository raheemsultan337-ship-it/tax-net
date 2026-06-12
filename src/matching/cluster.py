"""Union-find clustering of matched pairs into resolved entities."""

from __future__ import annotations


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_clusters(n_records: int,
                   match_pairs: list[tuple[int, int]]) -> list[list[int]]:
    uf = UnionFind(n_records)
    for i, j in match_pairs:
        uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n_records):
        groups.setdefault(uf.find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])
