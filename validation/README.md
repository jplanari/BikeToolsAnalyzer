# Validation tooling

Scripts that test how far the physics models can be trusted. They are not part of the app and are not imported by it.
Run them from the repository root. Output goes to `validation/output/` (not committed).

## 1. CdA error budget (simulation): `cda_error_budget.py`

```
python -m validation.cda_error_budget --runs 200 --seed 1
```

Simulates rides with a known CdA (0.300 m2), makes the inputs the estimator relies on wrong in controlled ways, and
measures the error of the fitted CdA: one error at a time (sign and size), then random errors alone and all together.
The result is the table of which inputs matter and the best accuracy that is possible at all.

The error sizes (`DEFAULT_SD` in `sim.py`) are assumptions. Replace them with the specifications of the power meter, the
gradient source and the weather source in use. A simulation shows how errors propagate; it does **not** show that the
estimator agrees with reality.

## 2. CdA self-consistency on real rides: `cda_consistency.py`

```
python -m validation.cda_consistency --user NAME [--db path/to/bikethools.db] [--since 2026-01-01] [--fetch] [--tags rides.csv]
```

Fits the CdA of each saved ride the way the app does, then checks what should not matter: spread between rides and per
bike, trends with wind, speed, gradient and temperature (Spearman, Bonferroni-corrected), the head-wind half against the
tail-wind half of each ride, and how much the weather wind changes the answer. Rides without stored weather are fitted in
still air and reported as such; `--fetch` downloads and stores their weather (the only thing that writes to the database).

It checks that the CdA behaves like a CdA. It cannot say it is right.

### Solo and group rides

Drafting lowers the CdA fitted into a head wind, so both reports split by ride type when they know it. Strava rides
imported from now on store the group size (`athlete_count`). For the rides already saved, make a template, fill in the
last column (`solo` or `group`; blank leaves a ride unlabelled) and pass it back:

```
python -m validation.cda_consistency --user NAME --tags-template rides.csv
python -m validation.cda_consistency --user NAME --tags rides.csv
python -m validation.wind_factor_sweep --user NAME --tags rides.csv
```

A label in the file wins over what the import recorded. Rows are matched on `ride_id`, or on `name` when there is none.

## 3. Wind height factor sweep: `wind_factor_sweep.py`

```
python -m validation.wind_factor_sweep --user NAME [--since 2026-01-01] [--factors 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] [--tags rides.csv]
```

The app scales the weather service's 10 m wind to rider height by `WIND_HEIGHT_FACTOR` (0.7). If it is wrong, the CdA
fitted on the head-wind stretches of a ride differs from the one fitted on its tail-wind stretches, though a rider's CdA
cannot depend on the wind direction. The sweep refits every ride for each factor, finds where the median head/tail ratio
reaches 1 (bootstrap interval over rides) and checks the factor on rides it was not fitted on (alternate rides, earlier
against later rides). It cannot separate drafting, direction-dependent shelter or wind errors that grow with speed:
the result is a hypothesis for the reference tests, not a calibration.

## 4. Reference tests (not scripted)

Precision on repeated runs, known position differences, and agreement with a wind tunnel or a velodrome analysis on a
calibrated power meter are field work. Record, for each run: the meter's zero offset, rider and kit mass, bike mass, tyres
and pressure, air temperature, pressure and humidity, and the wind. `cda_consistency.py` and `ride_metrics()` can
analyse those rides too.
