# BikeTools Analyzer

BikeTools Analyzer is a local, privacy-focused Python dashboard for analyzing cycling performance data. Built with Streamlit, it processes GPX files to provide advanced metrics, interactive maps, climb detection, and physics-based aerodynamic (CdA) estimation using historical weather data.

## Key Features

📊 Performance Metrics

- Power Analysis: Calculates Normalized Power (NP), Intensity Factor (IF), and Training Stress Score (TSS).
- Power Curve: Generates a Mean Maximal Power (MMP) curve to analyze best efforts across durations.
- Heart Rate: Automatic LTHR estimation (Friel method) and Time-in-Zone distributions.

💨 Aerodynamic Analysis (Beta)

- CdA Estimation: Estimates your Coefficient of Drag * Area (CdA) by solving the equation of motion.
- Weather Integration: Automatically fetches historical Wind Speed, Wind Direction, Temperature, and Pressure from the Open-Meteo API for the exact time and location of your ride.
- Vector Math: Calculates "Air Speed" (vs Ground Speed) by computing the headwind/tailwind component relative to the rider's compass bearing.

🏔️ Climb Detection

- Auto-Detection: Identifies climbs based on gradient and length thresholds (default >1km, >1%).
- Segmentation: Breaks climbs down into subsections to visualize gradient changes.
- VAM Calculation: Computes Vertical Ascent Speed (meters/hour).

🗺️ Visualization & History

- Interactive Maps: Folium-based route maps with highlighted climb segments.
- Local Database: Uses SQLite (bikethools.db) to store ride history and user profiles locally.
- User Profiles: Manage multiple rider profiles with specific FTP, Weight, and LTHR settings.

## Installation
Clone the repository: 
```
git clone [https://github.com/jplanari/BikeToolsAnalyzer.git](https://github.com/jplanari/BikeToolsAnalyzer.git)
cd bike-tools-analyzer
```
Install dependencies:
```
pip install -r requirements.txt
```

## Usage

- Run the application: 
``` 
streamlit run app.py
```

- Navigate the Dashboard:

* Sidebar: Create a user profile (set your Weight and FTP for accurate physics/metrics).

* Analyze Upload: Upload a .gpx file. The app will automatically parse it, fetch weather data, and generate the dashboard.

* Ride History: View a log of previously analyzed rides stored in the local database.

## Project Structure
```
bike-tools-analyzer/
├── app.py                  # Entry point
├── biketools_db.sqlite     # (Auto-generated)
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── analysis/           # Algorithms & Detection
    │   ├── __init__.py
    │   └── climbs.py
    ├── data/               # IO, Database, & Parsing
    │   ├── __init__.py
    │   ├── db.py
    │   ├── gpx.py          # (Was utilities.py)
    │   └── weather.py
    │   └── strava.py
    ├── physics/            # Core Science Logic
    │   ├── __init__.py
    │   ├── aerodyn.py
    │   ├── hr.py
    │   └── power.py
    └── viz/                # UI & Visualization
        ├── __init__.py
        ├── dashboard.py    # (Was ui.py)
        ├── maps.py         # (Was map_view.py)
        └── plots.py        # (Was graphical.py)
```
## The Physics Model
The application uses a forward-integration physics model to solve for Aerodynamic Drag ($F_{aero}$):

$P_{legs} \cdot (1 - Loss_{dt}) = P_{aero} + P_{roll} + P_{grav} + P_{accel} $

Where:

- $P_{aero}$: Derived from calculated Air Speed (Ground Speed + Wind Vector).
- Air Density ($\rho$): Calculated dynamically using Ideal Gas Law based on historical Temp/Pressure.
- $P_{roll} = C_{rr} \cdot Mass \cdot g \cdot v$ (Default $C_{rr} = 0.004$).
- $P_{grav} = m \cdot g \cdot v \cdot \sin(\text{slope})$.Note: CdA estimation requires power meter data and is most accurate during steady-state riding.

## How to Obtain Strava Session Cookies

To retrieve your Strava session cookies to upload GPX files directly from your Strava account, follow these steps:
1. Log in to your Strava account using a web browser (preferably Chrome or Firefox).
2. Open the Developer Tools (F12 or right-click and select "Inspect").:
3. Navigate to the "Application" tab (in Chrome) or "Storage" tab (in Firefox).
4. Under "Cookies," find and select the Strava domain (https://www.strava.com).
5. Look for the cookie named "strava_session" and copy its value.
6. Paste this value into the Strava Session Cookie field in the BikeTools Analyzer app when prompted.

## Requirements
Python 3.8 + 
```
streamlit
pandas
numpy
matplotlib
folium
streamlit-folium
gpxpy
geographiclib
scipy
requests
```

## License
MIT License. Feel free to modify and use for your personal training analysis.
