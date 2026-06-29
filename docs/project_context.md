# Project Context: Gold Key Neighborhood Traffic Simulation

This project simulates and analyzes relative traffic volume on road segments in the Gold Key residential neighborhood in Milford, PA, USA.

## Architecture & Design Decisions

### Bounding Box / Polygon
The neighborhood boundary is defined by a 15-point coordinate polygon:
`[(-74.9510518, 41.3246133), (-74.95221112, 41.3144843), (-74.9531335, 41.3051583), (-74.9516411, 41.2984673), (-74.9420818, 41.2961607), (-74.93352, 41.2984501), (-74.9322002, 41.3030443), (-74.9318243, 41.3053815), (-74.9359653, 41.3074506), (-74.93212, 41.3163538), (-74.9261524, 41.3277617), (-74.9199992, 41.3303436), (-74.9152089, 41.3401647), (-74.94572, 41.3459503), (-74.9510518, 41.3246133)]`

To ensure that the exit node (which lies on the boundary) is not cropped out, a spatial buffer of `0.001` degrees is applied to the polygon before querying OpenStreetMap.

### Ingestion & Spatial Setup
- **OSMnx (>=2.0.0)**: Used to download the drivable road network (`drive`).
- **Pike County GIS API Integration**: House locations are retrieved from the Pike County Address Points MapServer layer 1. To optimize performance and prevent repeated network requests, the fetched GeoJSON data is cached locally in `cache/gis_address_points.geojson`.
- **Conversion to Undirected**: OSMnx road networks are directed by default. We convert it to an undirected MultiGraph via `ox.convert.to_undirected` to treat roads as two-way.
- **Exit Intersection**: The intersection of Gold Key Road and Log Tavern Road. Found dynamically by locating a node that connects to both:
  - an edge with "gold key" in its name attribute
  - an edge with "log tavern" in its name attribute
- **Snapping**: House coordinates (Point geometries) are snapped to road network nodes based on their street address to prevent incorrect routing near parallel roads or intersections. Suffixes (e.g. `DR` -> `DRIVE`, `RD` -> `ROAD`) are normalized for both address points and OpenStreetMap edges. The point is snapped to the nearest node on the matching street; if no street match is found, it falls back to the nearest node on the entire graph. If the input data contains Polygon geometries (like OSM building footprints), centroids are calculated prior to snapping.

### Routing Algorithm
- Residents take the shortest physical path by distance (`length` attribute) to the exit.
- Dijkstra's shortest-path algorithm (`networkx.shortest_path`) is used.
- Parallel edges (MultiGraph structure) are resolved by selecting the edge with the shortest physical length.
- Traffic volume is incremented by 2 (1 trip out, 1 trip in) for each traversed edge.

### Normalization and Outputs
- Traffic volume is normalized to relative volume: $volume / max\_volume$ (0.0 to 1.0).
- Output products:
  - `output/traffic_map.png`: Heatmap visualization of relative traffic.
  - `output/house_connections.png`: Map visualization showing house locations (points/squares) and their snapped connection paths to the road network.
  - `output/traffic_volumes.csv`: Detailed CSV showing traffic volume and relative traffic per road segment.

## Project Structure
- [requirements.txt](../requirements.txt): Production requirements.
- [requirements-dev.txt](../requirements-dev.txt): Development requirements.
- [pytest.ini](../pytest.ini): Unit test configurations.
- [traffic_model.py](../traffic_model.py): Core simulation and visualization code.
- [test_traffic_model.py](../test_traffic_model.py): Unit test suite.
- `output/`: Folder containing the generated output products (git ignored).
