"""
Unit tests for the Gold Key neighborhood traffic simulation model.
"""

from typing import List, Tuple, Any
from pathlib import Path
import pytest
import requests
import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon, Point

from traffic_model import (
    create_buffered_polygon,
    find_exit_node,
    simulate_traffic,
    normalize_traffic,
    load_house_locations,
    normalize_street_name,
    extract_street_from_address,
    get_house_nodes,
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


def test_load_house_locations_from_cache(tmp_path: Path) -> None:
    """
    Test loading house locations from a cached GeoJSON file.
    """
    # Create dummy GeoJSON
    gdf_dummy = gpd.GeoDataFrame(
        [{"OBJECTID": 1, "SiteType": "R1", "PrimaryAddress": "123 Main St"}],
        geometry=[Point(-74.9, 41.3)],
        crs="EPSG:4326"
    )
    cache_file = tmp_path / "cache.geojson"
    gdf_dummy.to_file(str(cache_file), driver="GeoJSON")

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    gdf_loaded = load_house_locations(poly, cache_path=str(cache_file))

    assert len(gdf_loaded) == 1
    assert gdf_loaded.iloc[0]["PrimaryAddress"] == "123 Main St"


def test_load_house_locations_api_fetch(tmp_path: Path, monkeypatch: Any) -> None:
    """
    Test fetching house locations from the API when cache doesn't exist.
    """
    # Mock API response
    mock_response_data = {
        "features": [
            {
                "attributes": {"OBJECTID": 2, "SiteType": "R1", "PrimaryAddress": "456 Oak Rd"},
                "geometry": {"x": -74.95, "y": 41.31}
            }
        ]
    }

    class MockResponse:
        """
        Mock class for requests Response.
        """
        def __init__(self, data: Any) -> None:
            """
            Initialize MockResponse.
            """
            self._data = data

        def raise_for_status(self) -> None:
            """
            Mock raise_for_status.
            """

        def json(self) -> Any:
            """
            Mock json response.
            """
            return self._data

    def mock_get(_url: str, _params: Any = None, **_kwargs: Any) -> MockResponse:
        """
        Mock requests.get method.
        """
        return MockResponse(mock_response_data)

    monkeypatch.setattr(requests, "get", mock_get)

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    cache_file = tmp_path / "fetched.geojson"

    assert not cache_file.exists()
    gdf = load_house_locations(poly, cache_path=str(cache_file))

    assert len(gdf) == 1
    assert gdf.iloc[0]["PrimaryAddress"] == "456 Oak Rd"
    assert cache_file.exists()

    # Read it back to verify file contents
    gdf_cached = gpd.read_file(str(cache_file))
    assert len(gdf_cached) == 1
    assert gdf_cached.iloc[0]["PrimaryAddress"] == "456 Oak Rd"


def test_normalize_street_name() -> None:
    """
    Test the normalization of street names.
    """
    assert normalize_street_name("Northwynd Dr") == "NORTHWYND DRIVE"
    assert normalize_street_name("White Deer Rd") == "WHITE DEER ROAD"
    assert normalize_street_name("Byron Rd.") == "BYRON ROAD"
    assert normalize_street_name("Birches End Ln") == "BIRCHES END LANE"
    assert normalize_street_name("Crows Ct") == "CROWS COURT"
    assert normalize_street_name("") == ""


def test_extract_street_from_address() -> None:
    """
    Test extracting street name from a full primary address.
    """
    assert extract_street_from_address("120 Northwynd Dr") == "Northwynd Dr"
    assert extract_street_from_address("107 Schlage Rd") == "Schlage Rd"
    assert extract_street_from_address("NoNumber Rd") == "NoNumber Rd"
    assert extract_street_from_address("") == ""


def test_get_house_nodes_empty() -> None:
    """
    Test get_house_nodes on an empty GeoDataFrame.
    """
    graph = nx.MultiGraph()
    buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    assert not get_house_nodes(graph, buildings)


def test_get_house_nodes_snapping() -> None:
    """
    Test that get_house_nodes snaps to the street associated with the address.
    """
    # Create a small graph:
    # Node 1 connects to Node 2 (Gold Key Road)
    # Node 2 connects to Node 3 (Pom Pom Court)
    graph = nx.MultiGraph()
    graph.graph["crs"] = "EPSG:4326"
    # Coordinates for nodes
    graph.add_node(1, x=-74.9380, y=41.3060)
    graph.add_node(2, x=-74.9382, y=41.3065)
    graph.add_node(3, x=-74.9382, y=41.3075)

    graph.add_edge(1, 2, key=0, name="Gold Key Road")
    graph.add_edge(2, 3, key=0, name="Pom Pom Court")

    # House is closer to Node 3 (Pom Pom Court) but addressed to "155 Gold Key Road"
    # Point coords: close to Node 3, but we snap to Node 2 (Gold Key Road)
    house_pos = Point(-74.9382, 41.3074)

    buildings = gpd.GeoDataFrame(
        [{"PrimaryAddress": "155 Gold Key Road"}],
        geometry=[house_pos],
        crs="EPSG:4326"
    )

    node_ids = get_house_nodes(graph, buildings)
    assert node_ids == [2]

    # Another house with an unknown street should fall back to the closest node (Node 3)
    buildings_fallback = gpd.GeoDataFrame(
        [{"PrimaryAddress": "100 Unknown Rd"}],
        geometry=[house_pos],
        crs="EPSG:4326"
    )
    node_ids_fallback = get_house_nodes(graph, buildings_fallback)
    assert node_ids_fallback == [3]
