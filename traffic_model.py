"""
Traffic simulation model for the Gold Key neighborhood in Milford, PA, USA.

This script calculates relative traffic volumes on road segments by routing
trips from homes to the neighborhood exit.
"""

from typing import List, Tuple, Any
import csv
import os
from shapely.geometry import Polygon
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib
import matplotlib.pyplot as plt



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


def download_buildings(polygon: Polygon) -> gpd.GeoDataFrame:
    """
    Downloads building footprints within the polygon.

    Args:
        polygon: Bounding polygon.

    Returns:
        A GeoDataFrame containing the building footprints.
    """
    gdf = ox.features_from_polygon(polygon, tags={"building": True})
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("OSMnx did not return a GeoDataFrame")
    print(f"Number of buildings discovered in the polygon: {len(gdf)}")
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


def get_house_nodes(
    graph: nx.MultiGraph, buildings: gpd.GeoDataFrame
) -> List[int]:
    """
    Calculates building centroids and snaps them to the nearest graph nodes.

    Args:
        graph: The road network graph.
        buildings: GeoDataFrame of building footprints.

    Returns:
        List of snapped node IDs corresponding to each house.
    """
    if buildings.empty:
        return []

    projected_crs = buildings.estimate_utm_crs()
    centroids = buildings.to_crs(projected_crs).geometry.centroid.to_crs(buildings.crs)
    x_coords = centroids.x.tolist()
    y_coords = centroids.y.tolist()

    # Snap to nearest nodes using OSMnx
    node_ids = ox.nearest_nodes(graph, X=x_coords, Y=y_coords)
    return [int(nid) for nid in node_ids]


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

    print("Downloading building footprints from OpenStreetMap...")
    buildings = download_buildings(buffered_poly)

    print("Identifying exit node...")
    exit_node = find_exit_node(graph)

    print("Snapping building footprints to nearest road nodes...")
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

    print("Execution complete. Outputs generated successfully.")


if __name__ == "__main__":
    main()
