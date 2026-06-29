"""
Unit tests for the Gold Key neighborhood traffic simulation model.
"""

from typing import List, Tuple
import pytest
import networkx as nx
from shapely.geometry import Polygon

from traffic_model import (
    create_buffered_polygon,
    find_exit_node,
    simulate_traffic,
    normalize_traffic,
)


def test_create_buffered_polygon() -> None:
    """
    Test creating a buffered polygon from a list of coordinates.
    """
    coords: List[Tuple[float, float]] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    poly = create_buffered_polygon(coords, buffer_size=0.1)
    assert isinstance(poly, Polygon)
    # The buffered area should be larger than the original 1.0 unit square
    assert poly.area > 1.0


def test_find_exit_node_success() -> None:
    """
    Test finding the exit node where Gold Key and Log Tavern intersect.
    """
    graph = nx.MultiGraph()
    # Add nodes
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(3)

    # Node 2 connects to Gold Key (edge 1-2) and Log Tavern (edge 2-3)
    graph.add_edge(1, 2, key=0, name="Gold Key Road")
    graph.add_edge(2, 3, key=0, name="Log Tavern Road")

    exit_node = find_exit_node(graph)
    assert exit_node == 2


def test_find_exit_node_with_list_names() -> None:
    """
    Test finding the exit node when names are stored as lists in OSM attributes.
    """
    graph = nx.MultiGraph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(3)

    # Node 2 connects to Gold Key (edge 1-2) and Log Tavern (edge 2-3)
    graph.add_edge(1, 2, key=0, name=["Gold Key Road", "State Route 2009"])
    graph.add_edge(2, 3, key=0, name=["Log Tavern Road", "County Road"])

    exit_node = find_exit_node(graph)
    assert exit_node == 2


def test_find_exit_node_failure() -> None:
    """
    Test that find_exit_node raises ValueError when no intersection is found.
    """
    graph = nx.MultiGraph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(3)

    graph.add_edge(1, 2, key=0, name="Gold Key Road")
    graph.add_edge(2, 3, key=0, name="Some Other Road")

    with pytest.raises(ValueError, match="Could not find intersection node"):
        find_exit_node(graph)


def test_simulate_traffic_single_path() -> None:
    """
    Test traffic routing and accumulation over a simple path.
    """
    graph = nx.MultiGraph()
    # Path: 1 -> 2 -> 3 (Exit is 3)
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(3)

    graph.add_edge(1, 2, key=0, length=10.0)
    graph.add_edge(2, 3, key=0, length=15.0)

    # Run traffic simulation with 1 house at Node 1
    simulate_traffic(graph, house_nodes=[1], exit_node=3)

    # Each edge along the path (1-2 and 2-3) should have traffic_volume of 2 (1 out, 1 in)
    assert graph[1][2][0]["traffic_volume"] == 2
    assert graph[2][3][0]["traffic_volume"] == 2


def test_simulate_traffic_parallel_edges() -> None:
    """
    Test that traffic routing prefers the shorter of two parallel edges.
    """
    graph = nx.MultiGraph()
    # Path: 1 -> 2 (Exit is 2)
    # Two parallel edges between 1 and 2:
    # Edge A: key=0, length=5.0
    # Edge B: key=1, length=10.0
    graph.add_node(1)
    graph.add_node(2)

    graph.add_edge(1, 2, key=0, length=5.0)
    graph.add_edge(1, 2, key=1, length=10.0)

    simulate_traffic(graph, house_nodes=[1], exit_node=2)

    # Traffic should route along the shorter edge (key=0)
    assert graph[1][2][0]["traffic_volume"] == 2
    assert graph[1][2][1]["traffic_volume"] == 0


def test_normalize_traffic() -> None:
    """
    Test normalization of traffic volumes.
    """
    graph = nx.MultiGraph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(3)

    graph.add_edge(1, 2, key=0, traffic_volume=10)
    graph.add_edge(2, 3, key=0, traffic_volume=4)

    max_vol = normalize_traffic(graph)

    assert max_vol == 10.0
    assert graph[1][2][0]["relative_traffic"] == 1.0
    assert graph[2][3][0]["relative_traffic"] == 0.4
