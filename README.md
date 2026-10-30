# BikeTools Analyzer

BikeTools Analyzer is a local, privacy-focused Python dashboard for analyzing cycling performance data. Built with Streamlit, it processes GPX files to provide advanced metrics, interactive maps, climb detection, and physics-based aerodynamic (CdA) estimation using historical weather data.

## Desktop app

A standalone version with its own window (no terminal, no Python install) can be built for Windows, macOS and Linux: see [desktop/README.md](desktop/README.md). Your data is kept in a per-user folder, separate from the program.

## Key Features

📊 Performance Metrics

- Power Analysis: Calculates Normalized Power (NP), Intensity Factor (IF), and Training Stress Score (TSS).
- Power Curve: Generates a Mean Maximal Power (MMP) curve to analyze best efforts across durations.
- Heart Rate: Automatic LTHR estimation (Friel method) and Time-in-Zone distributions.

🏋️ Training page

- **Load:** fitness / fatigue / form (PMC) with a **forecast from your planned sessions**, ramp rate, acute:chronic ratio and monotony with plain-language status, weekly or monthly volume (hours, km, TSS, kJ, elevation).
- **Power profile:** best power curves for the last 42 / 90 days, the season and all time; 2- and 3-parameter critical power (CP, W′, Pmax); CP and FTP history; W/kg and ×FTP profile; records with the date they were set.
- **Durability:** how much of your fresh power you keep after 1, 2 and 3 MJ of work.
- **Intensity:** weekly/monthly easy / moderate / hard distribution (power or HR) and the polarization index.
- **PDF report:** the "📄 PDF report" box at the top builds a 6-page A4 report (summary and load chart, volume and ramp, power profile and CP, durability and records, intensity, recent rides and planned sessions) for the period you pick.

📐 Segment analysis (bottom of the ride page)

- Pick any stretch of the ride with a distance slider, or choose a **lap** (FIT files), and get the numbers a whole ride gets: time, distance, ascent, gradient, VAM, average speed, average power and W/kg, normalized power, IF, TSS, work, VI, average HR, EF and average cadence (while pedalling), with the power / HR / speed / cadence traces of that stretch.
- Stretches of 300 m or more get their gradient profile and, if they average 1 % or more, the same performance estimator as climbs. *Compare all laps* tabulates every lap side by side.

⛰️ Climb performance estimator (Analyze Upload → Detected Climbs → View Details)

- For each detected climb, change **system weight, CdA, rolling resistance, air density and power** and see the time it would take, or ask for the **power needed for a target time**. A steady-state model on the climb's gradient profile (no wind); the *as ridden* line applies it to your actual power so you can see how close it gets, and a small table shows what −1 kg, −0.01 m² CdA, −0.001 Crr and +10 W are worth.

🔬 Deeper analytics (Training page → Deeper analytics)

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
- Local Database: Uses SQLite (bikethools.db) to store ride history and user profiles locally. Original ride files are kept compressed in their own table, and the database is copied to `backups/` (newest 3 kept) before "Update ride history" and before the one-time file migration.
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
    │   ├── correlations.py   # power vs HR / cadence analytics
    │   ├── durability.py
    │   ├── intensity.py      # intensity distribution, polarization index
    │   ├── power.py
    │   ├── power_profile.py  # window curves, CP models, history
    │   ├── ride_metrics.py   # per-ride VI, EF, decoupling, zones, histograms
    │   └── training_load.py  # PMC, forecast, ramp, ACWR, monotony, volume
    ├── data/               # IO, Database, & Parsing
    │   ├── db.py
    │   ├── fit.py
    │   ├── gpx.py
    │   ├── pipeline.py       # Shared parse → 1 Hz → metrics pipeline
    │   ├── strava.py         # Strava API client (OAuth, streams)
    │   ├── history.py        # Recompute stored rides
    │   ├── strava_import.py  # Single-ride and history import
    │   ├── wahoo.py          # Wahoo Cloud API client (OAuth, workouts, FIT download)
    │   ├── wahoo_import.py   # Wahoo single-ride, history and sync
    │   └── weather.py
    ├── physics/            # Core Science Logic
    │   ├── aerodyn.py
    │   └── thermal.py
    └── viz/                # UI & Visualization
        ├── dashboard.py        # entry points used by app.py (re-exports the pages)
        ├── sidebar.py          # navigation, user profile, import, Strava connection
        ├── ride_analysis.py    # Analyze Upload page
        ├── user_corner.py      # User Corner: profile summary and ride-history update
        ├── climb_estimator.py  # per-climb what-if estimator (weight, CdA, Crr, power)
        ├── deeper_analytics.py # Training page section: power-HR, power-cadence, recovery/efficiency
        ├── report_pdf.py       # PDF training report (matplotlib)
        ├── training_page.py    # Training page (load, power profile, durability, intensity, planning)
        ├── formatting.py       # shared display helpers
        ├── sections.py         # tab-like sections that render only the selected one
        ├── maps.py
        ├── plots.py
        └── training_plots.py
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

## Connecting Wahoo

Rides can be synced straight from the Wahoo Cloud API (OAuth). Unlike Strava it provides the **original FIT file** recorded by your ELEMNT, so power, heart rate and cadence keep their full resolution and nothing is rebuilt from resampled streams.

**One-time setup**
1. Register an app at <https://developers.wahooligan.com> and add `http://localhost:8501/` as a callback URL. New apps start in sandbox mode (25 requests per 5 minutes, 100 per hour); a large history import simply pauses and can be run again, or you can ask Wahoo to move the app to production.
2. Add its **Client ID** and **Client Secret** to `.streamlit/secrets.toml` (see `secrets.toml.example`):
   ```toml
   [wahoo]
   client_id = "..."
   client_secret = "..."
   redirect_uri = "http://localhost:8501/"
   ```
3. Restart the app.

**Any ride, not just the recent ones:** under the ride list, **🔎 Find a ride by date** (Strava and Wahoo) lists every ride between two dates; pick one and *Analyze ride*. **Clear** returns to the recent list.

**Usage** (sidebar → Data Import → *Wahoo*): **Connect Wahoo**, then *Recent rides → Analyze ride*, **Sync new rides**, or *Import ride history*, exactly as for Strava. Files are stored as `wahoo_<id>.fit`. A ride that starts within 2 minutes of one you already have (uploaded, or imported from Strava) is skipped, so if you imported the same rides from Strava earlier, delete those first if you want the full-resolution FIT versions instead.

**Sending planned sessions to your ELEMNT.** Sessions built from intervals on the Training page get a **📲 Send to Wahoo** button (once Wahoo is connected). It creates a workout plan (fixed-power steps with a ±3 % band, repeats for your reps, rest only between repeats) and schedules it on the session's date at the start time you choose, so it appears in the Wahoo app and on the head unit. **↻ Update on Wahoo** resends changes, **Remove from Wahoo** (or deleting the session) takes it off. Wahoo only delivers sessions from today to 6 days ahead, and quick TSS entries have no steps, so they can't be sent. This needs the `plans_read`, `plans_write` and `workouts_write` scopes; if you connected before, disconnect and connect again. Optionally set `timezone = "Europe/Madrid"` in `[wahoo]` (default: this computer's zone).

Notes: Wahoo does not document its workout types, so only type 0 (outdoor cycling) is imported; add other ids to `CYCLING_WORKOUT_TYPES` in `src/data/wahoo.py` if your indoor rides are skipped. Wahoo refresh tokens are single-use and are re-saved on every refresh.

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
