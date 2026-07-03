"""
Traffic simulation model for the Gold Key neighborhood in Milford, PA, USA.

This script calculates relative traffic volumes on road segments by routing
trips from homes to the neighborhood exit.
"""

from typing import List, Tuple, Any, Optional, NamedTuple
import csv
import json
import os
import re
import requests
from shapely.geometry import Polygon, Point, LineString
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class RoadEdge(NamedTuple):
    """
    NamedTuple representing a snapped road segment in the network.
    """
    u: int
    v: int
    key: int


class ConnectionLinesResult(NamedTuple):
    """
    NamedTuple representing the output of _build_connection_lines.
    """
    points: List[Any]
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]]



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
    # Ensure OSMnx parses access and barrier tags on both nodes and ways
    for tag in ["access", "barrier"]:
        if tag not in ox.settings.useful_tags_node:
            ox.settings.useful_tags_node.append(tag)
        if tag not in ox.settings.useful_tags_way:
            ox.settings.useful_tags_way.append(tag)

    # Download graph without simplifying to preserve intermediate nodes with barriers
    graph = ox.graph_from_polygon(polygon, network_type="drive", simplify=False)
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("OSMnx did not return a MultiDiGraph")

    # Simplify the graph while explicitly preserving barrier nodes (e.g. gates)
    simplified_graph = ox.simplify_graph(graph, node_attrs_include=["barrier"])
    if not isinstance(simplified_graph, nx.MultiDiGraph):
        raise TypeError("OSMnx simplification did not return a MultiDiGraph")
    return simplified_graph


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


def find_matching_street_edges(
    graph: nx.MultiGraph, street_name: str
) -> List[RoadEdge]:
    """
    Finds all edges in the graph matching the given street name.

    Args:
        graph: The road network graph.
        street_name: The street name to match.

    Returns:
        A list of RoadEdge NamedTuples matching the street name.
    """
    norm_s = normalize_street_name(street_name)
    norm_s_ns = normalize_street_name_no_space(street_name)

    exact_edges = []
    partial_edges = []

    for u, v, key, data in graph.edges(keys=True, data=True):
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
                exact_edges.append(RoadEdge(int(u), int(v), int(key)))
                break
            if (
                norm_s in norm_osm
                or norm_osm in norm_s
                or norm_s_ns in norm_osm_ns
                or norm_osm_ns in norm_s_ns
            ):
                partial_edges.append(RoadEdge(int(u), int(v), int(key)))
                break

    return exact_edges if exact_edges else partial_edges


def get_house_edges(
    graph: nx.MultiGraph, buildings: gpd.GeoDataFrame
) -> List[RoadEdge]:
    """
    Calculates building centroids or uses point geometries and snaps them
    to graph edges. If 'PrimaryAddress' is present, snaps to edges associated
    with that street name; otherwise, snaps to the nearest edge globally.

    Args:
        graph: The road network graph.
        buildings: GeoDataFrame of building footprints or address points.

    Returns:
        List of snapped RoadEdge NamedTuples corresponding to each house.
    """
    if buildings.empty:
        return []

    # Check if all geometries are points
    if (buildings.geometry.geom_type == "Point").all():
        points = buildings.geometry.tolist()
    else:
        points = (
            buildings.to_crs(buildings.estimate_utm_crs())
            .geometry.centroid.to_crs(buildings.crs)
            .tolist()
        )

    edge_ids: List[RoadEdge] = []
    has_address = "PrimaryAddress" in buildings.columns

    for idx, point in enumerate(points):
        snapped_edge = None
        if has_address:
            addr = buildings.iloc[idx]["PrimaryAddress"]
            if addr and isinstance(addr, str):
                street = extract_street_from_address(addr)
                candidates = find_matching_street_edges(graph, street)
                if candidates:
                    sub_graph = graph.edge_subgraph(candidates)
                    res = ox.nearest_edges(sub_graph, X=point.x, Y=point.y)
                    snapped_edge = RoadEdge(int(res[0]), int(res[1]), int(res[2]))

        # Fallback to nearest edge globally
        if snapped_edge is None:
            res = ox.nearest_edges(graph, X=point.x, Y=point.y)
            snapped_edge = RoadEdge(int(res[0]), int(res[1]), int(res[2]))

        edge_ids.append(snapped_edge)

    return edge_ids


def _get_closer_endpoint(graph: nx.MultiGraph, edge: RoadEdge, exit_node: int) -> Optional[int]:
    """
    Finds which of the two endpoints of a RoadEdge (u or v) is closer to the exit node.
    Returns None if both are unreachable.
    """
    try:
        dist_u = nx.shortest_path_length(graph, source=edge.u, target=exit_node, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist_u = float("inf")

    try:
        dist_v = nx.shortest_path_length(graph, source=edge.v, target=exit_node, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist_v = float("inf")

    if dist_u == float("inf") and dist_v == float("inf"):
        return None
    return edge.u if dist_u <= dist_v else edge.v


def _find_shortest_edge_key(edges_between: Any) -> Any:
    """
    Finds the key of the edge with the shortest physical length.
    """
    best_key = None
    min_length = float("inf")
    for key_id, edge_data in edges_between.items():
        length = float(edge_data.get("length", float("inf")))
        if length < min_length:
            min_length = length
            best_key = key_id
    return best_key


def simulate_traffic(
    graph: nx.MultiGraph, house_edges: List[RoadEdge], exit_node: int
) -> None:
    """
    Runs routing from each house's snapped edge to the exit node
    and increments the traffic volume of the traversed edges.

    Args:
        graph: The road network graph.
        house_edges: List of snapped house RoadEdge NamedTuples.
        exit_node: The exit node ID.
    """
    # Initialize traffic_volume to 0 for all edges
    for _, _, data in graph.edges(data=True):
        data["traffic_volume"] = 0

    # Route and aggregate traffic
    for edge in house_edges:
        source_node = _get_closer_endpoint(graph, edge, exit_node)
        if source_node is None:
            continue

        try:
            path = nx.shortest_path(
                graph, source=source_node, target=exit_node, weight="length"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        # Increment traffic volume by 2 for the home edge itself
        graph[edge.u][edge.v][edge.key]["traffic_volume"] += 2

        # Increment traffic volume by 2 for each edge in the shortest path
        for node_a, node_b in zip(path[:-1], path[1:]):
            edges_between = graph[node_a][node_b]
            best_key = _find_shortest_edge_key(edges_between)
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
    points: List[Any],
    house_edges: List[RoadEdge],
    graph_proj: nx.MultiGraph,
) -> ConnectionLinesResult:
    """
    Builds segment coordinate pairs for house connection lines, projecting each
    house point onto the nearest point on its snapped road segment.

    Args:
        points: List of Shapely points representing house locations.
        house_edges: List of snapped RoadEdge NamedTuples.
        graph_proj: Projected road network graph.

    Returns:
        A ConnectionLinesResult NamedTuple containing the list of points and connection lines.
    """
    filtered_points = []
    lines = []

    for idx, geom in enumerate(points):
        edge = house_edges[idx]

        # Order-agnostic check in projected graph
        found_u, found_v = None, None
        if graph_proj.has_edge(edge.u, edge.v, edge.key):
            found_u, found_v = edge.u, edge.v
        elif graph_proj.has_edge(edge.v, edge.u, edge.key):
            found_u, found_v = edge.v, edge.u

        if found_u is not None and found_v is not None:
            edge_data = graph_proj.edges[found_u, found_v, edge.key]

            # Retrieve or construct edge geometry
            if "geometry" in edge_data:
                edge_geom = edge_data["geometry"]
            else:
                edge_geom = LineString([
                    (graph_proj.nodes[found_u]["x"], graph_proj.nodes[found_u]["y"]),
                    (graph_proj.nodes[found_v]["x"], graph_proj.nodes[found_v]["y"])
                ])

            # Project point onto edge to get the nearest point on the road segment
            snapped_pt = edge_geom.interpolate(edge_geom.project(geom))
            lines.append(((geom.x, geom.y), (snapped_pt.x, snapped_pt.y)))
            filtered_points.append(geom)

    return ConnectionLinesResult(filtered_points, lines)


def plot_house_connections(
    graph: nx.MultiGraph,
    buildings: gpd.GeoDataFrame,
    house_edges: List[RoadEdge],
    filename: str,
) -> None:
    """
    Generates a map visualization showing house locations connected to
    their snapped road segments and saves it as an image.

    Args:
        graph: The road network graph.
        buildings: GeoDataFrame containing the house address points.
        house_edges: List of snapped RoadEdge NamedTuples corresponding to each house.
        filename: Destination image filepath.
    """
    if buildings.empty or not house_edges:
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

    # Build connection lines between each house and its snapped road segment
    result = _build_connection_lines(points, house_edges, graph_proj)

    if result.lines:
        ax.add_collection(
            LineCollection(
                result.lines,
                colors="#ffffff",
                linestyles="--",
                linewidths=0.5,
                alpha=0.4,
                zorder=1,
            )
        )

    # Plot houses as small squares/rectangles
    if result.points:
        ax.scatter(
            [pt.x for pt in result.points],
            [pt.y for pt in result.points],
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
    Removes nodes/edges representing closed or gated roads that cannot carry
    traffic by identifying OSM barrier or access restrictions on graph nodes.

    Args:
        graph: The road network graph.
    """
    nodes_to_remove = []
    for node, data in graph.nodes(data=True):
        barrier = data.get("barrier")
        access = data.get("access")

        # Check if node is gated/blocked or restricted to private use only
        if barrier == "gate" or access in ["private", "no"]:
            nodes_to_remove.append(node)

    if nodes_to_remove:
        print(f"Removing {len(nodes_to_remove)} gated/restricted nodes from the network...")
        for node in nodes_to_remove:
            # Print affected roads for log visibility
            incident_edges = list(graph.edges(node, data=True))
            street_names = set()
            for _, _, data in incident_edges:
                name_attr = data.get("name")
                if name_attr:
                    names = name_attr if isinstance(name_attr, list) else [name_attr]
                    street_names.update(names)
            print(
                f"Removing Node {node} (gated/restricted) "
                f"affecting roads: {', '.join(street_names)}"
            )
            graph.remove_node(node)


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

    print("Snapping house locations to nearest road edges...")
    house_edges = get_house_edges(graph, buildings)

    print("Running traffic simulation...")
    simulate_traffic(graph, house_edges, exit_node)

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
        house_edges,
        os.path.join("output", "house_connections.png"),
    )

    print("Execution complete. Outputs generated successfully.")


if __name__ == "__main__":
    main()
