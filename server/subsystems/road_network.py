"""Road network subsystem: graph, routing, congestion, disruptions."""

import heapq
from typing import Dict, List, Optional, Tuple


class Edge:
    def __init__(self, from_node: str, to_node: str, base_time: int):
        self.from_node = from_node
        self.to_node = to_node
        self.base_time = base_time
        self.congestion_multiplier: float = 1.0
        self.disrupted: bool = False
        self.disruption_type: Optional[str] = None
        self.disruption_remaining: int = 0

    def effective_time(self) -> float:
        if self.disrupted and self.disruption_type == "road_closure":
            return float("inf")
        return self.base_time * self.congestion_multiplier

    def tick(self) -> None:
        """Advance disruption timers by 1 minute."""
        if self.disrupted and self.disruption_remaining > 0:
            self.disruption_remaining -= 1
            if self.disruption_remaining == 0:
                self.disrupted = False
                self.disruption_type = None
                self.congestion_multiplier = 1.0


class RoadNetwork:
    """City road network with routing, congestion, and disruption support."""

    def __init__(self):
        from .constants import CITY_NODES, CITY_EDGES, CONGESTION_CURVES, EPISODE_START_HOUR

        self.node_ids: List[str] = [n["id"] for n in CITY_NODES]
        self._edges: Dict[str, Edge] = {}
        self._congestion_curves: dict = CONGESTION_CURVES
        self._episode_start_hour: int = EPISODE_START_HOUR
        self._step_count: int = 0  # tracks elapsed minutes for TOD
        self._last_congestion_hour: Optional[int] = None  # cache: skip recompute if hour unchanged

        for edge in CITY_EDGES:
            key = self._edge_key(edge["from"], edge["to"])
            self._edges[key] = Edge(edge["from"], edge["to"], edge["base_time"])

        # Apply initial TOD congestion
        self._apply_tod_congestion()

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def edges(self) -> Dict[str, Edge]:
        """Return a copy of the edge dictionary to prevent external mutation."""
        return dict(self._edges)

    def get_travel_time(self, from_node: str, to_node: str) -> float:
        """Get effective travel time between two adjacent nodes."""
        edge = self._get_edge(from_node, to_node)
        if edge is None:
            return float("inf")
        return edge.effective_time()

    def shortest_path(self, from_node: str, to_node: str) -> List[str]:
        """
        Dijkstra shortest path. Returns list of node IDs or empty list if unreachable.
        road_closure edges return inf and are skipped.
        """
        if from_node == to_node:
            return [from_node]

        pq: List[Tuple[float, str, List[str]]] = [(0.0, from_node, [from_node])]
        visited: set = set()

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node == to_node:
                return path
            for neighbor, edge in self._neighbors(node):
                if neighbor not in visited and edge.effective_time() < float("inf"):
                    new_cost = cost + edge.effective_time()
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

        return []  # Unreachable

    def route_travel_time(self, route: List[str]) -> int:
        """Total minutes for a precomputed route."""
        total = 0.0
        for i in range(len(route) - 1):
            t = self.get_travel_time(route[i], route[i + 1])
            if t == float("inf"):
                return 999
            total += t
        return int(total)

    def set_disruption(
        self,
        from_node: str,
        to_node: str,
        disruption_type: str,
        remaining_steps: int,
    ) -> bool:
        """Apply a disruption to an edge. Returns True if applied."""
        edge = self._get_edge(from_node, to_node)
        if edge is None:
            return False
        edge.disrupted = True
        edge.disruption_type = disruption_type
        edge.disruption_remaining = remaining_steps
        if disruption_type == "accident":
            edge.congestion_multiplier = 3.0
        elif disruption_type == "road_closure":
            edge.congestion_multiplier = float("inf")
        return True

    def clear_disruption(self, from_node: str, to_node: str) -> None:
        """Remove disruption from an edge."""
        edge = self._get_edge(from_node, to_node)
        if edge:
            edge.disrupted = False
            edge.disruption_type = None
            edge.disruption_remaining = 0
            edge.congestion_multiplier = 1.0

    def tick(self) -> None:
        """Advance all edge disruptions by 1 minute and update TOD congestion."""
        self._step_count += 1
        self._apply_tod_congestion()
        for edge in self._edges.values():
            edge.tick()

    def _apply_tod_congestion(self) -> None:
        """
        Recompute congestion multipliers for all edges based on current time of day.
        Disruptions (accident/road_closure) override TOD multipliers.
        """
        from .constants import interpolate_congestion

        hour = (self._episode_start_hour + self._step_count // 60) % 24
        frac_hour = hour + (self._step_count % 60) / 60.0  # fractional hour

        for key, edge in self._edges.items():
            # Skip edges with active disruptions — disruptions set their own multipliers
            if edge.disrupted:
                continue
            # Try both node IDs as curve keys
            curve = self._congestion_curves.get(edge.from_node, [])
            multiplier = interpolate_congestion(curve, frac_hour)
            edge.congestion_multiplier = multiplier

    def get_active_disruptions(self) -> List[Dict]:
        """Return list of currently disrupted edges."""
        result = []
        for key, edge in self._edges.items():
            if edge.disrupted:
                result.append({
                    "edge_key": key,
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "type": edge.disruption_type,
                    "remaining_steps": edge.disruption_remaining,
                })
        return result

    # =========================================================================
    # Internal
    # =========================================================================

    def _edge_key(self, from_node: str, to_node: str) -> str:
        nodes = sorted([from_node, to_node])
        return f"{nodes[0]}->{nodes[1]}"

    def _get_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        key = self._edge_key(from_node, to_node)
        return self._edges.get(key)

    def _neighbors(self, node: str) -> List[Tuple[str, Edge]]:
        """Return (neighbor_node, edge) for all edges adjacent to node."""
        result = []
        for edge in self._edges.values():
            if edge.from_node == node:
                result.append((edge.to_node, edge))
            elif edge.to_node == node:
                result.append((edge.from_node, edge))
        return result
