"""
Traffic simulation model for the Gold Key neighborhood in Milford, PA, USA.

This script calculates relative traffic volumes on road segments by routing
trips from homes to the neighborhood exit.
"""

from typing import List, Tuple, Any
import csv
import json
import os
import re
import requests
from shapely.geometry import Polygon, Point
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# Suffix normalization dictionary for mapping address street endings to OSM style
SUFFIX_MAP = {
    "DR": "DRIVE",
    "RD": "ROAD",
    "LN": "LANE",
    "CT": "COURT",
    "TER": "TERRACE",
    "PL": "PLACE",
    "ST": "STREET",
    "AVE": "AVENUE",
    "AV": "AVENUE",
    "BLVD": "BOULEVARD",
    "PKWY": "PARKWAY",
    "HWY": "HIGHWAY",
    "WY": "WAY",
}


def normalize_street_name(name: str) -> str:
    """
    Normalizes a street name by converting to uppercase, cleaning spaces,
    removing non-alphanumeric characters, and expanding standard abbreviations.

    Args:
        name: The raw street name.

    Returns:
        The normalized street name.
    """
    if not name:
        return ""
    # Convert to uppercase
    name = name.strip().upper()
    # Replace non-alphanumeric with spaces
    name = re.sub(r"[^A-Z0-9\s]", " ", name)
    # Collapse multiple spaces
    name = " ".join(name.split())

    tokens = name.split()
    if not tokens:
        return ""

    # Check if last token is an abbreviation to map
    if tokens[-1] in SUFFIX_MAP:
        tokens[-1] = SUFFIX_MAP[tokens[-1]]

    return " ".join(tokens)


def normalize_street_name_no_space(name: str) -> str:
    """
    Normalizes a street name and removes all whitespace. This allows matching
    names with spacing differences (e.g. "BLUE JAY" vs "BLUEJAY").

    Args:
        name: The raw street name.

    Returns:
        The normalized street name without any spaces.
    """
    return normalize_street_name(name).replace(" ", "")


def extract_street_from_address(address: str) -> str:
    """
    Extracts the street name component from a primary address string by
    removing the leading house/street number if present.

    Args:
        address: The primary address (e.g. "120 NORTHWYND DR").

    Returns:
        The street name portion.
    """
    if not address:
        return ""
    parts = address.split()
    if len(parts) > 1:
        # Check if the first token is a house number (numeric or digit-leading)
        first = parts[0]
        if first.isdigit() or (first[:-1].isdigit() and first[-1].isalpha()):
            return " ".join(parts[1:])
    return address


def create_buffered_polygon(
    coords: List[Tuple[float, float]], buffer_size: float = 0.001
) -> Polygon:
    """
    Creates a shapely Polygon from the coordinates and applies a buffer.

    Args:
        coords: List of (longitude, latitude) tuples.
        buffer_size: Buffer size in degrees.

    Returns:
        A buffered shapely Polygon.
    """
    poly = Polygon(coords)
    buffered_poly = poly.buffer(buffer_size)
    if not isinstance(buffered_poly, Polygon):
        raise TypeError("Buffer operation did not return a Polygon")
    return buffered_poly


def download_drive_graph(polygon: Polygon) -> nx.MultiDiGraph:
    """
    Downloads the drivable road network within the polygon.

    Args:
        polygon: Bounding polygon.

    Returns:
        A NetworkX MultiDiGraph representing the road network.
    """
    graph = ox.graph_from_polygon(polygon, network_type="drive")
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("OSMnx did not return a MultiDiGraph")
    return graph


def convert_to_undirected(graph: nx.MultiDiGraph) -> nx.MultiGraph:
    """
    Converts the directed graph to an undirected graph.

    Args:
        graph: Directed MultiDiGraph.

    Returns:
        Undirected MultiGraph.
    """
    undirected_graph = ox.convert.to_undirected(graph)
    if not isinstance(undirected_graph, nx.MultiGraph):
        raise TypeError("Conversion did not return a MultiGraph")
    return undirected_graph


def load_house_locations(
    polygon: Polygon, cache_path: str = "cache/gis_address_points.geojson"
) -> gpd.GeoDataFrame:
    """
    Loads house locations within the polygon. First checks if a cached GeoJSON
    file exists. If not, fetches the data from the Pike County GIS API,
    saves it to the cache file, and returns it.

    Args:
        polygon: Bounding polygon.
        cache_path: Path to the cached GeoJSON file.

    Returns:
        A GeoDataFrame containing the house address points.
    """
    if os.path.exists(cache_path):
        gdf = gpd.read_file(cache_path)
        print(f"Loaded {len(gdf)} address points from cache: {cache_path}")
        return gdf

    print("Fetching address points from Pike County GIS API...")
    features: List[dict] = []
    offset = 0

    while True:
        try:
            url = (
                "https://gis.pikepa.org/arcgis/rest/services/"
                "PikeCo_AddressPoints/MapServer/1/query"
            )
            resp = requests.get(
                url,
                params={
                    "where": "SiteType = 'R1'",
                    "geometry": json.dumps({
                        "rings": [list(polygon.exterior.coords)],
                        "spatialReference": {"wkid": 4326}
                    }),
                    "geometryType": "esriGeometryPolygon",
                    "inSR": "4326",
                    "spatialRel": "esriSpatialRelIntersects",
                    "outSR": "4326",
                    "outFields": "OBJECTID,SiteType,PrimaryAddress",
                    "returnGeometry": "true",
                    "resultOffset": str(offset),
                    "resultRecordCount": "1000",
                    "f": "json"
                },
                verify=True,
                timeout=15
            )
            resp.raise_for_status()
            batch = resp.json().get("features", [])
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to query Pike County GIS API: {e}") from e

        if not batch:
            break
        features.extend(batch)
        if len(batch) < 1000:
            break
        offset += len(batch)

    print(f"Retrieved {len(features)} residential address points from Pike County GIS.")

    # Convert features to a GeoDataFrame using list comprehensions to reduce local variables
    valid_feats = [
        f for f in features
        if f.get("geometry") and "x" in f["geometry"] and "y" in f["geometry"]
    ]
    gdf = gpd.GeoDataFrame(
        [f.get("attributes", {}) for f in valid_feats],
        geometry=[Point(f["geometry"]["x"], f["geometry"]["y"]) for f in valid_feats],
        crs="EPSG:4326"
    )

    # Create the cache directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf.to_file(cache_path, driver="GeoJSON")
    print(f"Saved {len(gdf)} address points to cache: {cache_path}")

    return gdf


def find_exit_node(graph: nx.MultiGraph) -> int:
    """
    Finds the node at the intersection of Gold Key Road and Log Tavern Road.

    A node is identified as the exit if it connects to at least one edge
    containing 'gold key' in its name and at least one edge containing
    'log tavern' in its name.

    Args:
        graph: The undirected road network graph.

    Returns:
        The integer node ID of the exit intersection.
    """
    for node in graph.nodes():
        has_gold_key = False
        has_log_tavern = False

        # Iterate over all incident edges
        for _, _, data in graph.edges(node, data=True):
            name_attr = data.get("name")
            if name_attr is None:
                continue

            # Normalize to list of strings
            names = (
                name_attr if isinstance(name_attr, list) else [name_attr]
            )
            for name in names:
                if not isinstance(name, str):
                    continue
                name_lower = name.lower()
                if "gold key road" in name_lower:
                    has_gold_key = True
                if "log tavern road" in name_lower:
                    has_log_tavern = True

        if has_gold_key and has_log_tavern:
            print(f"Located exit node: {node}")
            return int(node)

    raise ValueError(
        "Could not find intersection node of Gold Key Road and Log Tavern Road."
    )


def find_matching_street_nodes(
    graph: nx.MultiGraph, street_name: str
) -> List[int]:
    """
    Finds all nodes in the graph that are endpoints of edges matching the
    given street name (either exact or partial match).

    Args:
        graph: The road network graph.
        street_name: The street name to match.

    Returns:
        A list of node IDs matching the street name.
    """
    norm_s = normalize_street_name(street_name)
    norm_s_ns = normalize_street_name_no_space(street_name)

    exact_nodes = set()
    partial_nodes = set()

    for u, v, data in graph.edges(data=True):
        name_attr = data.get("name")
        if name_attr is None:
            continue
        names = name_attr if isinstance(name_attr, list) else [name_attr]
        for name in names:
            if not isinstance(name, str):
                continue
            norm_osm = normalize_street_name(name)
            norm_osm_ns = normalize_street_name_no_space(name)

            if norm_s == norm_osm or norm_s_ns == norm_osm_ns:
                exact_nodes.add(u)
                exact_nodes.add(v)
            elif (
                norm_s in norm_osm
                or norm_osm in norm_s
                or norm_s_ns in norm_osm_ns
                or norm_osm_ns in norm_s_ns
            ):
                partial_nodes.add(u)
                partial_nodes.add(v)

    nodes = exact_nodes if exact_nodes else partial_nodes
    return list(nodes)


def get_house_nodes(
    graph: nx.MultiGraph, buildings: gpd.GeoDataFrame
) -> List[int]:
    """
    Calculates building centroids or uses point geometries and snaps them
    to graph nodes. If 'PrimaryAddress' is present, snaps to nodes associated
    with that street name; otherwise, snaps to the nearest node globally.

    Args:
        graph: The road network graph.
        buildings: GeoDataFrame of building footprints or address points.

    Returns:
        List of snapped node IDs corresponding to each house.
    """
    if buildings.empty:
        return []

    # Check if all geometries are points
    geom_types = buildings.geometry.geom_type.unique()
    if len(geom_types) == 1 and geom_types[0] == "Point":
        points = buildings.geometry.tolist()
    else:
        projected_crs = buildings.estimate_utm_crs()
        centroids = buildings.to_crs(projected_crs).geometry.centroid.to_crs(buildings.crs)
        points = centroids.tolist()

    node_ids: List[int] = []
    has_address = "PrimaryAddress" in buildings.columns

    for idx, point in enumerate(points):
        snapped_id = None
        if has_address:
            addr = buildings.iloc[idx]["PrimaryAddress"]
            if addr and isinstance(addr, str):
                street = extract_street_from_address(addr)
                candidates = find_matching_street_nodes(graph, street)
                if candidates:
                    sub_graph = graph.subgraph(candidates)
                    snapped_id = int(ox.nearest_nodes(sub_graph, X=point.x, Y=point.y))

        # Fallback to nearest node globally
        if snapped_id is None:
            snapped_id = int(ox.nearest_nodes(graph, X=point.x, Y=point.y))

        node_ids.append(snapped_id)

    return node_ids


def simulate_traffic(
    graph: nx.MultiGraph, house_nodes: List[int], exit_node: int
) -> None:
    """
    Runs Dijkstra's shortest path routing from each house to the exit node
    and increments the traffic volume of the traversed edges.

    Args:
        graph: The road network graph.
        house_nodes: List of snapped house node IDs.
        exit_node: The exit node ID.
    """
    # Initialize traffic_volume to 0 for all edges
    for _, _, data in graph.edges(data=True):
        data["traffic_volume"] = 0

    # Route and aggregate traffic
    for house in house_nodes:
        try:
            path = nx.shortest_path(
                graph, source=house, target=exit_node, weight="length"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        # Increment traffic volume by 2 for each edge in the path
        for u, v in zip(path[:-1], path[1:]):
            edges_between = graph[u][v]
            # Handle parallel edges by choosing the shortest one
            def get_edge_length(key_id: Any, eb: Any = edges_between) -> float:
                return float(eb[key_id].get("length", float("inf")))
            best_key = min(edges_between.keys(), key=get_edge_length)
            edges_between[best_key]["traffic_volume"] += 2


def normalize_traffic(graph: nx.MultiGraph) -> float:
    """
    Computes relative traffic volume for each edge normalized by the max volume.

    Args:
        graph: The road network graph.

    Returns:
        The maximum traffic volume in the network.
    """
    max_volume = 0.0
    for _, _, data in graph.edges(data=True):
        vol = float(data.get("traffic_volume", 0))
        max_volume = max(max_volume, vol)

    for _, _, data in graph.edges(data=True):
        if max_volume > 0:
            data["relative_traffic"] = (
                float(data.get("traffic_volume", 0)) / max_volume
            )
        else:
            data["relative_traffic"] = 0.0

    return max_volume


def save_traffic_to_csv(graph: nx.MultiGraph, filename: str) -> None:
    """
    Saves the road network traffic data to a CSV file.

    Args:
        graph: The road network graph.
        filename: Destination filepath.
    """
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "u",
                "v",
                "key",
                "street_name",
                "traffic_volume",
                "relative_traffic",
            ]
        )
        for u, v, k, data in graph.edges(keys=True, data=True):
            name_attr = data.get("name", "Unnamed")
            if isinstance(name_attr, list):
                name = "; ".join(name_attr)
            else:
                name = str(name_attr)
            writer.writerow(
                [
                    u,
                    v,
                    k,
                    name,
                    data.get("traffic_volume", 0),
                    data.get("relative_traffic", 0.0),
                ]
            )


def plot_traffic_heatmap(graph: nx.MultiGraph, filename: str) -> None:
    """
    Generates a traffic heatmap visualization and saves it as an image.

    Args:
        graph: The road network graph.
        filename: Destination image filepath.
    """
    # Project to UTM to ensure correct aspect ratio and north-up conformal orientation
    graph_proj = ox.project_graph(graph)

    edge_colors: List[Any] = []
    edge_widths: List[float] = []
    colormap = matplotlib.colormaps["plasma"]

    for _, _, data in graph_proj.edges(data=True):
        rel_t = float(data.get("relative_traffic", 0.0))
        color: Any
        if rel_t == 0.0:
            # Slate-grey for zero-travel roads
            color = (0.22, 0.25, 0.3, 1.0)
            width = 0.8
        else:
            # Shift colormap input range to [0.2, 1.0] to avoid dark colors
            color = colormap(0.2 + 0.8 * rel_t)
            # Scale linewidth from 1.2 to 6.0 based on relative traffic
            width = 1.2 + 4.8 * rel_t

        edge_widths.append(width)
        edge_colors.append(color)

    # Plot using OSMnx with a deep slate background
    fig, _ = ox.plot_graph(
        graph_proj,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        node_size=0,
        bgcolor="#0c0f12",
        show=False,
        close=False,
    )
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_connection_lines(
    points: List[Point],
    house_nodes: List[int],
    graph_proj: nx.MultiGraph,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Builds segment coordinate pairs for house connection lines.

    Args:
        points: List of shapely Point objects representing house locations.
        house_nodes: List of snapped node IDs.
        graph_proj: Projected road network graph.

    Returns:
        A list of coordinate pairs representing connection lines.
    """
    lines = []
    for idx, geom in enumerate(points):
        if idx >= len(house_nodes):
            break
        node_id = house_nodes[idx]
        if node_id in graph_proj.nodes:
            node_data = graph_proj.nodes[node_id]
            lines.append(((geom.x, geom.y), (node_data["x"], node_data["y"])))
    return lines


def plot_house_connections(
    graph: nx.MultiGraph,
    buildings: gpd.GeoDataFrame,
    house_nodes: List[int],
    filename: str,
) -> None:
    """
    Generates a map visualization showing house locations connected to
    their snapped road nodes and saves it as an image.

    Args:
        graph: The road network graph.
        buildings: GeoDataFrame containing the house address points.
        house_nodes: List of snapped node IDs corresponding to each house.
        filename: Destination image filepath.
    """
    if buildings.empty or not house_nodes:
        print("No house locations to plot connections for.")
        return

    # Project the graph to UTM to ensure correct aspect ratio
    graph_proj = ox.project_graph(graph)

    # Project buildings to the same CRS
    buildings_proj = buildings.to_crs(graph_proj.graph["crs"])

    # Plot road network using OSMnx with a deep slate background
    fig, ax = ox.plot_graph(
        graph_proj,
        edge_color="#2c3238",  # Slate-grey/dark-grey for roads
        edge_linewidth=1.0,
        node_size=0,
        bgcolor="#0c0f12",  # Sleek dark background
        show=False,
        close=False,
    )

    # Extract coordinates based on geometry type
    if (buildings_proj.geometry.geom_type == "Point").all():
        points = buildings_proj.geometry.tolist()
    else:
        points = buildings_proj.geometry.centroid.tolist()

    # Build connection lines between each house and its snapped road node
    lines = _build_connection_lines(points, house_nodes, graph_proj)

    if lines:
        ax.add_collection(
            LineCollection(
                lines,
                colors="#ffffff",
                linestyles="--",
                linewidths=0.5,
                alpha=0.4,
                zorder=1,
            )
        )

    # Plot houses as small squares/rectangles
    ax.scatter(
        [pt.x for pt in points],
        [pt.y for pt in points],
        color="#ff6b6b",  # Vibrant coral/red
        s=8,
        marker="s",
        label="House Locations",
        zorder=2,
    )

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def remove_closed_roads(graph: nx.MultiGraph) -> None:
    """
    Removes edges representing closed or gated roads that cannot carry traffic.
    Currently overrides:
    - 'Wood Place': gated and closed to traffic.

    Args:
        graph: The road network graph.
    """
    closed_names = {"wood place"}
    edges_to_remove = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        name_attr = data.get("name")
        if name_attr:
            names = name_attr if isinstance(name_attr, list) else [name_attr]
            if any(
                isinstance(n, str) and n.lower() in closed_names
                for n in names
            ):
                edges_to_remove.append((u, v, k))

    if edges_to_remove:
        print(f"Removing {len(edges_to_remove)} edges for locally overridden closed roads...")
        for u, v, k in edges_to_remove:
            graph.remove_edge(u, v, k)


def main() -> None:
    """
    Main execution flow for downloading data, running the model,
    and outputting the results.
    """
    print("Initializing Gold Key neighborhood geometry...")
    coords = [
        (-74.9510518, 41.3246133),
        (-74.95221112, 41.3144843),
        (-74.9531335, 41.3051583),
        (-74.9516411, 41.2984673),
        (-74.9420818, 41.2961607),
        (-74.93352, 41.2984501),
        (-74.9322002, 41.3030443),
        (-74.9318243, 41.3053815),
        (-74.9359653, 41.3074506),
        (-74.93212, 41.3163538),
        (-74.9261524, 41.3277617),
        (-74.9199992, 41.3303436),
        (-74.9152089, 41.3401647),
        (-74.94572, 41.3459503),
        (-74.9510518, 41.3246133),
    ]

    buffered_poly = create_buffered_polygon(coords, buffer_size=0.001)

    print("Downloading street network from OpenStreetMap...")
    dir_graph = download_drive_graph(buffered_poly)
    graph = convert_to_undirected(dir_graph)
    remove_closed_roads(graph)

    print("Loading house locations...")
    buildings = load_house_locations(buffered_poly)

    print("Identifying exit node...")
    exit_node = find_exit_node(graph)

    print("Snapping house locations to nearest road nodes...")
    house_nodes = get_house_nodes(graph, buildings)

    print("Running traffic simulation...")
    simulate_traffic(graph, house_nodes, exit_node)

    print("Normalizing traffic volumes...")
    max_vol = normalize_traffic(graph)
    print(f"Simulation completed. Max edge traffic volume: {max_vol}")

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    print("Saving traffic data to output/traffic_volumes.csv...")
    save_traffic_to_csv(graph, os.path.join("output", "traffic_volumes.csv"))

    print("Saving traffic heatmap to output/traffic_map.png...")
    plot_traffic_heatmap(graph, os.path.join("output", "traffic_map.png"))

    print("Saving house connection visualization to output/house_connections.png...")
    plot_house_connections(
        graph,
        buildings,
        house_nodes,
        os.path.join("output", "house_connections.png"),
    )

    print("Execution complete. Outputs generated successfully.")


if __name__ == "__main__":
    main()
