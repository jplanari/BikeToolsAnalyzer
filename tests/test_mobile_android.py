"""The Android project is generated in CI; these check the edits patch.py makes, without an Android SDK."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "mobile" / "android"
spec = importlib.util.spec_from_file_location("android_patch", ROOT / "patch.py")
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name">
        <activity android:name=".MainActivity">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
            </intent-filter>
        </activity>
    </application>
</manifest>
"""
GRADLE = """apply plugin: 'com.android.application'

android {
    namespace "io.github.jplanari.biketools"
    defaultConfig {
        versionCode 1
        versionName "1.0"
    }
    buildTypes {
        release {
            minifyEnabled false
        }
    }
}
"""


def test_manifest_gets_the_pairing_link_and_cleartext_once():
    out = patch.patch_manifest(MANIFEST)
    assert out.count('android:scheme="biketools"') == 1 and 'android:host="pair"' in out
    assert 'android:usesCleartextTraffic="true"' in out
    assert out.index("biketools") < out.index("</activity>")
    assert patch.patch_manifest(out) == out                      # running it twice changes nothing


def test_gradle_version_and_signing():
    out = patch.patch_gradle(GRADLE, "2.3.4", 17)
    assert 'versionName "2.3.4"' in out and "versionCode 17" in out and "signingConfigs" not in out
    signed = patch.patch_gradle(GRADLE, "2.3.4", 17, sign=True)
    assert "signingConfigs" in signed and "signingConfig signingConfigs.release" in signed
    assert 'storePassword System.getenv("KEYSTORE_PASSWORD")' in signed          # secrets come from the environment, never the file
    assert patch.patch_gradle(signed, "2.3.4", 17, sign=True).count("signingConfigs {") == 1
    with pytest.raises(ValueError):
        patch.patch_gradle("android {}", "1", 1)


def test_app_name():
    assert "BikeTools" in patch.patch_strings('<string name="app_name">Old</string><string name="title_activity_main">Old</string>')


def test_icons_are_written_in_every_density(tmp_path):
    pytest.importorskip("PIL")
    patch.write_icons(tmp_path)
    for dpi in ("mdpi", "hdpi", "xhdpi", "xxhdpi", "xxxhdpi"):
        assert (tmp_path / f"mipmap-{dpi}" / "ic_launcher.png").stat().st_size > 200
        assert (tmp_path / f"mipmap-{dpi}" / "ic_launcher_foreground.png").exists()


def test_capacitor_config_is_consistent():
    cfg = json.loads((ROOT / "capacitor.config.json").read_text())
    assert cfg["webDir"] == "../web" and (ROOT / cfg["webDir"] / "index.html").exists()
    assert cfg["plugins"]["CapacitorHttp"]["enabled"] is True and cfg["server"]["cleartext"] is True
    pkg = json.loads((ROOT / "package.json").read_text())["dependencies"]
    assert {"@capacitor/core", "@capacitor/android", "@capacitor/cli", "@capacitor/app"} <= set(pkg)
    majors = {v.lstrip("^~").split(".")[0] for v in pkg.values()}
    assert majors == {"6"}                                       # plugins and core must share a major version


def test_workflow_publishes_on_mobile_tags():
    text = (ROOT.parents[1] / ".github" / "workflows" / "android.yml").read_text()
    assert "mobile-v*" in text and "gh release create" in text and "ANDROID_KEYSTORE_BASE64" in text
