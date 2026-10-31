"""
The database layer, split by subject. Everything is re-exported here, so `from src.data import db` and
`db.save_ride(...)` work as they always did. Settings that tests or launchers change (DB_FILE, ...) are
forwarded to the submodule that reads them."""
import sys
import time
import types

from .core import (BACKUP_KEEP, BUSY_TIMEOUT_S, DB_FILE, _add_column, _connect, _date_filter, _enable_wal, _parse_dates, backup_db, connect, data_stamp)
from .users import (DEFAULT_FTP, DEFAULT_LTHR, DEFAULT_W_PRIME, delete_user, get_all_users, get_user_data, save_user, set_auto_thresholds, set_user_thresholds, set_user_timezone, set_user_w_prime, update_user_crank, update_user_ftp, update_user_physics)
from .rides import (METRICS_VERSION, _METRIC_COLUMNS, _NEEDS_UPDATE_SQL, _do_move_ride_files, _insert_children, _iso_timestamp, _metrics_fields, _migration_lock, _move_ride_files, _opt_float, _opt_int, _pack, _unpack, build_ride_metadata, count_rides_needing_update, get_hr20, get_ride_file, get_ride_start_times, get_ride_zones, get_rides_for_recompute, get_rides_without_hr20, get_user_best_power, get_user_rides, get_user_tss_history, save_ride, set_ride_hr20, strava_id_from_filename, update_ride_metrics)
from .stats import (get_hist, get_kj_records, get_power_records, get_ride_stats, get_zone_rows, period_coverage)
from .auth import (OAUTH_STATE_TTL_S, _SECRET_COLUMNS, _decrypt_fields, consume_oauth_state, create_oauth_state, delete_app_credentials, delete_polar_auth, delete_strava_auth, delete_wahoo_auth, delete_wahoo_sent, encrypt_plaintext_secrets, get_app_credentials, get_imported_polar_ids, get_imported_strava_ids, get_imported_wahoo_ids, get_polar_auth, get_strava_auth, get_wahoo_auth, get_wahoo_sent, latest_strava_ride_date, latest_wahoo_ride_date, polar_id_from_filename, save_app_credentials, save_polar_auth, save_strava_auth, save_wahoo_auth, save_wahoo_sent, wahoo_id_from_filename)
from .weekly import (claim_weekly_email, delete_smtp_settings, finish_weekly_email, get_smtp_settings, get_weekly_email, release_weekly_email, save_smtp_settings, save_weekly_email, users_with_weekly_email)
from .planning import (add_structured_session, delete_planned_session, get_planned_sessions, replace_planned_sessions)
from .bikes import (_BIKE_FIELDS, _BIKE_PARTS, _bike_values, delete_bike, get_bikes, get_ride_bike, save_bike, set_default_bike, set_ride_bike, update_bike)
from .ride_weather import (get_ride_weather, save_ride_weather)
from .schema import (MIGRATIONS, _has_rides, _init_db, _init_lock, _initialised, _migration_per_user_file_hash, _migration_query_indexes, _migration_ride_weather, _migration_utc_dates, init_db, migrate_db, run_migrations, schema_version)
from . import core, users, rides, stats, auth, weekly, planning, bikes, ride_weather, schema
_SUBMODULES = (core, users, rides, stats, auth, weekly, planning, bikes, ride_weather, schema)


class _DbModule(types.ModuleType):
    """Setting a name here (monkeypatch.setattr(db, "DB_FILE", ...)) also sets it where it is used."""
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for sub in _SUBMODULES:
            if name in sub.__dict__:
                setattr(sub, name, value)


sys.modules[__name__].__class__ = _DbModule
