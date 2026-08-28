from collections import deque
from typing import Iterable


def build_adjacency(num_nodes: int, edge_index: Iterable[Iterable[int]], directed: bool = False) -> list[set[int]]:
    adjacency = [set() for _ in range(num_nodes)]
    for edge in edge_index:
        source, target = (int(value) for value in edge)
        if source == target:
            continue
        adjacency[source].add(target)
        if not directed:
            adjacency[target].add(source)
    return adjacency


def bfs_distances(adjacency: list[set[int]], source: int) -> tuple[list[int | None], list[int | None]]:
    distances: list[int | None] = [None] * len(adjacency)
    parents: list[int | None] = [None] * len(adjacency)
    distances[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if distances[neighbor] is not None:
                continue
            distances[neighbor] = int(distances[node]) + 1
            parents[neighbor] = node
            queue.append(neighbor)
    return distances, parents


def shortest_path(adjacency: list[set[int]], source: int, target: int) -> list[int]:
    distances, parents = bfs_distances(adjacency, source)
    if distances[target] is None:
        return []
    path = [target]
    while path[-1] != source:
        parent = parents[path[-1]]
        if parent is None:
            return []
        path.append(parent)
    path.reverse()
    return path


def bfs_frontiers(adjacency: list[set[int]], source: int, max_depth: int) -> list[list[int]]:
    distances, _ = bfs_distances(adjacency, source)
    return [
        [node for node, distance in enumerate(distances) if distance == depth]
        for depth in range(max_depth + 1)
    ]
