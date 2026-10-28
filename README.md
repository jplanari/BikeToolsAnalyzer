# BikeTools Analyzer

BikeTools Analyzer is a local, privacy-focused Python dashboard for analyzing cycling performance data. Built with Streamlit, it processes GPX files to provide advanced metrics, interactive maps, climb detection, and physics-based aerodynamic (CdA) estimation using historical weather data.

## Key Features

📊 Performance Metrics

- Power Analysis: Calculates Normalized Power (NP), Intensity Factor (IF), and Training Stress Score (TSS).
- Power Curve: Generates a Mean Maximal Power (MMP) curve to analyze best efforts across durations.
- Heart Rate: Automatic LTHR estimation (Friel method) and Time-in-Zone distributions.

🔬 Deeper Analytics (User Corner)

- Pick a period and compare it with the previous one of the same length.
- Power ↔ Heart rate: time heatmap, HR at each power, correlation and slope, weekly HR at 65 % FTP (falling = fitter).
- Power ↔ Cadence: heatmap, cadence by power zone, Coggan quadrant analysis around your own median cadence (set your crank length there), weekly median cadence.
- Recovery & efficiency: heart-rate recovery 60 s after hard efforts, efficiency factor and aerobic decoupling on steady endurance rides.
- Computed from compact per-ride histograms stored at save time; older rides get them via **Update ride history**.

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

* Analyze Upload: Upload a .gpx/.fit file, or pick a ride from Strava (see *Connecting Strava*). The app will automatically parse it, fetch weather data, and generate the dashboard.

* Ride History: View a log of previously analyzed rides stored in the local database.

## Project Structure
```
BikeToolsAnalyzer/
├── app.py                  # Entry point
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example  # Strava API credentials template
└── src/
    ├── analysis/           # Algorithms & Detection
    │   ├── climbs.py
    │   ├── hr.py
    │   ├── power.py
    │   └── training_load.py  # PMC (CTL/ATL/TSB)
    ├── data/               # IO, Database, & Parsing
    │   ├── db.py
    │   ├── fit.py
    │   ├── gpx.py
    │   ├── pipeline.py       # Shared parse → 1 Hz → metrics pipeline
    │   ├── strava.py         # Strava API client (OAuth, streams)
    │   ├── strava_import.py  # Single-ride and history import
    │   └── weather.py
    ├── physics/            # Core Science Logic
    │   ├── aerodyn.py
    │   └── thermal.py
    └── viz/                # UI & Visualization
        ├── dashboard.py
        ├── maps.py
        └── plots.py
```
## The Physics Model
The application uses a forward-integration physics model to solve for Aerodynamic Drag ($F_{aero}$):

$P_{legs} \cdot (1 - Loss_{dt}) = P_{aero} + P_{roll} + P_{grav} + P_{accel} $

Where:

- $P_{aero}$: Derived from calculated Air Speed (Ground Speed + Wind Vector).
- Air Density ($\rho$): Calculated dynamically using Ideal Gas Law based on historical Temp/Pressure.
- $P_{roll} = C_{rr} \cdot Mass \cdot g \cdot v$ (Default $C_{rr} = 0.004$).
- $P_{grav} = m \cdot g \cdot v \cdot \sin(\text{slope})$.Note: CdA estimation requires power meter data and is most accurate during steady-state riding.

## Connecting Strava

Rides can be imported straight from Strava through the official API (OAuth). Each rider profile connects its own Strava account.

**One-time setup**
1. Go to <https://www.strava.com/settings/api> and create an API application.
2. Set **Authorization Callback Domain** to `localhost`.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (it is git-ignored) and fill in your app's **Client ID** and **Client Secret**:
   ```toml
   [strava]
   client_id = "12345"
   client_secret = "your-client-secret"
   redirect_uri = "http://localhost:8501/"
   ```
4. Restart the app.

**Usage** (sidebar → Data Import → *Strava*)
- **Connect Strava**: approve access in the tab that opens. The app asks for `activity:read_all` so private ("Only You") rides can be imported too. Tokens are stored per profile in the local database and refreshed automatically.
- **Recent rides → Analyze ride**: downloads the ride and opens the full analysis (it is saved to your history like an upload).
- **Import ride history**: imports every ride since a date into the history (PMC, power records). Rides already imported, and rides that start within 2 minutes of a file you uploaded yourself, are skipped, so TSS is never counted twice. Strava allows 100 read requests per 15 minutes and 1,000 per day; a large import pauses before the limit, and running it again later resumes where it stopped.
- **Disconnect Strava** revokes the app's access and deletes the stored tokens.

Notes:
- The API does not provide the original GPX/FIT file, so imported rides are rebuilt from Strava's data streams (time, GPS, altitude, heart rate, power, cadence, temperature) and stored as `strava_<id>.json`.
- TSS for imported rides is computed with the profile's **current** FTP.
- Strava's API terms only allow showing an athlete's data to that athlete, so keep this as a personal tool rather than a shared dashboard.

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
