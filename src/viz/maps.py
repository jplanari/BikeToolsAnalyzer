# src/map_view.py
import folium

def create_route_map(df, climbs=None):
    """
    Generates a Folium map with the route (blue) and highlighted climbs (red).
    """
    # Basic validation
    if df.empty or 'lat' not in df.columns or 'lon' not in df.columns:
        return None

    # Filter out invalid coordinates (0.0 or NaN)
    valid_points = df[(df['lat'].notna()) & (df['lon'].notna()) & (df['lat'] != 0)]
    
    if valid_points.empty:
        return None

    # Calculate center of the map
    center_lat = valid_points['lat'].mean()
    center_lon = valid_points['lon'].mean()

    # Initialize Map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    # --- OPTIMIZATION: DOWNSAMPLE ---
    # Browsers struggle with >5000 points. We target ~1000 points for the blue line.
    # This does NOT affect the statistics, only the blue line visual.
    total_points = len(valid_points)
    stride = max(1, total_points // 1000)


    # 1. Draw the Full Route (Blue)
    route_coords = valid_points[['lat', 'lon']].values.tolist()
    folium.PolyLine(
        route_coords,
        color="#3388ff",
        weight=4,
        opacity=0.7,
        tooltip="Full Route"
    ).add_to(m)

    # 2. Highlight Climbs (Red)
    if climbs:
        for i, c in enumerate(climbs):
            # Extract the segment for this climb
            segment = df.loc[c['start_idx']:c['end_idx']]
            climb_coords = segment[['lat', 'lon']].values.tolist()
            
            # Create a popup text with stats
            popup_txt = (
                f"<b>Climb #{i+1}</b><br>"
                f"Length: {c['length_m']/1000:.2f} km<br>"
                f"Avg Grad: {c['avg_gradient_pct']:.1f}%<br>"
                f"Gain: {int(c['elev_gain_m'])}m"
            )

            # Draw the climb line
            folium.PolyLine(
                climb_coords,
                color="red",
                weight=6,
                opacity=0.9,
                popup=folium.Popup(popup_txt, max_width=200),
                tooltip=f"Climb #{i+1}"
            ).add_to(m)

            # Add a marker at the summit
            summit = segment.iloc[-1]
            folium.CircleMarker(
                location=[summit['lat'], summit['lon']],
                radius=5,
                color="red",
                fill=True,
                fill_color="white",
                tooltip=f"Summit #{i+1}"
            ).add_to(m)

    # Fit bounds to show the whole route
    m.fit_bounds(m.get_bounds())

    return m
