"""Adjusts the Android project that `npx cap add android` generates: the biketools:// pairing link, the version, the
launcher icon and (when a keystore is supplied) release signing. Run from mobile/android after `npx cap sync`.

    VERSION_NAME=1.2.0 VERSION_CODE=12 [KEYSTORE_PATH=… KEYSTORE_PASSWORD=… KEY_ALIAS=… KEY_PASSWORD=…] python patch.py

The edits are plain string functions so tests can check them without an Android SDK."""
import os
import re
import sys
from pathlib import Path

ICON_BG = (15, 111, 143)
LINK_FILTER = """
            <intent-filter>
                <action android:name="android.intent.action.VIEW" />
                <category android:name="android.intent.category.DEFAULT" />
                <category android:name="android.intent.category.BROWSABLE" />
                <data android:scheme="biketools" android:host="pair" />
            </intent-filter>
"""
SIGNING = """
    signingConfigs {
        release {
            storeFile file(System.getenv("KEYSTORE_PATH"))
            storePassword System.getenv("KEYSTORE_PASSWORD")
            keyAlias System.getenv("KEY_ALIAS")
            keyPassword System.getenv("KEY_PASSWORD")
        }
    }
"""


def patch_manifest(xml):
    """Adds the pairing link to the main activity and allows plain-http servers on a home network."""
    if "android:scheme=\"biketools\"" not in xml:
        i = xml.index("</activity>")
        xml = xml[:i] + LINK_FILTER.strip("\n") + "\n        " + xml[i:]
    if "usesCleartextTraffic" not in xml:
        xml = xml.replace("<application", '<application\n        android:usesCleartextTraffic="true"', 1)
    return xml


def patch_gradle(text, version_name, version_code, sign=False):
    text, n1 = re.subn(r"versionCode\s+\d+", f"versionCode {int(version_code)}", text, count=1)
    text, n2 = re.subn(r'versionName\s+"[^"]*"', f'versionName "{version_name}"', text, count=1)
    if not (n1 and n2):
        raise ValueError("versionCode/versionName not found in app/build.gradle")
    if sign and "signingConfigs" not in text:
        text, n = re.subn(r"(\n\s*android\s*\{)", lambda m: m.group(1) + SIGNING, text, count=1)
        text, n3 = re.subn(r"(buildTypes\s*\{\s*release\s*\{)", r"\1\n            signingConfig signingConfigs.release", text, count=1)
        if not (n and n3):
            raise ValueError("could not add signing to app/build.gradle")
    return text


def patch_strings(xml, name="BikeTools"):
    xml = re.sub(r"(<string name=\"app_name\">)[^<]*(</string>)", rf"\g<1>{name}\g<2>", xml)
    return re.sub(r"(<string name=\"title_activity_main\">)[^<]*(</string>)", rf"\g<1>{name}\g<2>", xml)


def draw_icon(size, background=True, scale=1.0):
    """A simple bicycle in white on the brand colour; `scale` < 1 keeps it inside an adaptive icon's safe zone."""
    from PIL import Image, ImageDraw
    big = size * 4
    img = Image.new("RGBA", (big, big), ICON_BG + (255,) if background else (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = big / 512 * scale
    o = big * (1 - scale) / 2

    def P(x, y):
        return (o + x * u, o + y * u)

    w = max(2, int(30 * u))
    for cx in (150, 362):
        r = 72
        d.ellipse([*P(cx - r, 320 - r), *P(cx + r, 320 + r)], outline="white", width=w)
    for a, b in (((150, 320), (220, 190)), ((220, 190), (322, 190)), ((322, 190), (362, 320)), ((220, 190), (262, 320)),
                 ((262, 320), (150, 320)), ((262, 320), (322, 190)), ((196, 150), (244, 150))):
        d.line([P(*a), P(*b)], fill="white", width=w, joint="curve")
    return img.resize((size, size), Image.LANCZOS)


def write_icons(res):
    sizes = {"mdpi": (48, 108), "hdpi": (72, 162), "xhdpi": (96, 216), "xxhdpi": (144, 324), "xxxhdpi": (192, 432)}
    for dpi, (legacy, adaptive) in sizes.items():
        folder = Path(res) / f"mipmap-{dpi}"
        folder.mkdir(parents=True, exist_ok=True)
        draw_icon(legacy).save(folder / "ic_launcher.png")
        draw_icon(legacy).save(folder / "ic_launcher_round.png")
        draw_icon(adaptive, background=False, scale=0.6).save(folder / "ic_launcher_foreground.png")
    bg = Path(res) / "values" / "ic_launcher_background.xml"
    if bg.parent.exists():
        bg.write_text('<?xml version="1.0" encoding="utf-8"?>\n<resources>\n    <color name="ic_launcher_background">#0F6F8F</color>\n</resources>\n')


def main(root="android"):
    root = Path(root)
    app = root / "app"
    version_name = os.environ.get("VERSION_NAME", "1.0.0")
    version_code = os.environ.get("VERSION_CODE", "1")
    sign = bool(os.environ.get("KEYSTORE_PATH"))
    manifest = app / "src/main/AndroidManifest.xml"
    manifest.write_text(patch_manifest(manifest.read_text()))
    gradle = app / "build.gradle"
    gradle.write_text(patch_gradle(gradle.read_text(), version_name, version_code, sign))
    strings = app / "src/main/res/values/strings.xml"
    if strings.exists():
        strings.write_text(patch_strings(strings.read_text()))
    write_icons(app / "src/main/res")
    print(f"patched: version {version_name} ({version_code}), signing {'release' if sign else 'debug'}")


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2])))
