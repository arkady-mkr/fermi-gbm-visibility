"""Check Fermi GBM visibility at a UTC time and a J2000 position."""

import argparse
from datetime import date, timedelta
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import urlencode
from urllib.request import urlopen
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits, votable
from astropy.time import Time, TimeDelta
from astropy.utils.exceptions import AstropyUserWarning
from scipy.optimize import brentq

ARCHIVE = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily"
TAP = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
EPOCH = Time("2001-01-01T00:00:00", scale="utc")
EARTH_RADIUS_M = 6_371_000
# POSHIST is sampled at about 1 Hz. Do not interpolate across missing samples.
MAX_ORBIT_STEP_S = 1.5
DEFAULT_CACHE = Path(__file__).resolve().parent / "fermi/data/gbm-visibility"


class ArchiveIndex(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.links = []
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)


def utc(met):
    return (EPOCH + TimeDelta(met, format="sec")).utc.isot + "Z"


def archive_file(day, product, cache, local_dirs, offline):
    base = f"{ARCHIVE}/{day:%Y/%m/%d}/current/"
    extension = "fit" if product == "poshist_all" else "pha"
    pattern = re.compile(rf"glg_{product}_{day:%y%m%d}_v(\d+)\.{extension}(?:\.gz)?")

    def latest(names):
        matches = [name for name in names if pattern.fullmatch(name)]
        return max(matches, key=lambda name: int(pattern.fullmatch(name)[1]), default=None)

    for directory in [*local_dirs, cache]:
        if directory.is_dir():
            name = latest(path.name for path in directory.iterdir() if path.is_file())
            if name:
                return directory / name, base + name
    if offline:
        raise FileNotFoundError(f"{product}, {day}: нет локального файла")

    with urlopen(base, timeout=30) as response:
        name = latest(ArchiveIndex(response.read().decode("utf-8")).links)
    if name is None:
        raise FileNotFoundError(f"{product}, {day}: файл отсутствует в {base}")

    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / name
    url = base + name
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        download = Path(tmp) / name
        with urlopen(url, timeout=60) as response, download.open("wb") as output:
            shutil.copyfileobj(response, output)
        # Check the FITS structure and available checksums before publishing to cache.
        with warnings.catch_warnings():
            warnings.simplefilter("error", AstropyUserWarning)
            with fits.open(download, checksum=True) as hdus:
                hdus.verify("exception")
        download.replace(destination)
    return destination, url


def read_orbit(path):
    with fits.open(path) as hdus:
        table = hdus["GLAST POS HIST"].data
        return (
            np.array(table["SCLK_UTC"], dtype=float),
            np.column_stack([table[f"POS_{axis}"] for axis in "XYZ"]),
            (table["FLAGS"] & 2) != 0,
        )


def orbit_check(met, window, source, orbits):
    times, positions, saa = orbits[0]
    for extra_times, extra_positions, extra_saa in orbits[1:]:
        keep = (extra_times < times[0]) | (extra_times > times[-1])
        times = np.r_[times, extra_times[keep]]
        positions = np.concatenate([positions, extra_positions[keep]])
        saa = np.r_[saa, extra_saa[keep]]
        order = np.argsort(times)
        times, positions, saa = times[order], positions[order], saa[order]

    def geometry(at):
        coords = np.array([np.interp(at, times, axis) for axis in positions.T])
        distance = np.linalg.norm(coords, axis=0)
        separation = np.rad2deg(np.arccos(np.clip(-source @ coords / distance, -1, 1)))
        radius = np.rad2deg(np.arcsin(EARTH_RADIUS_M / distance))
        return separation, radius, separation - radius

    breaks = np.flatnonzero(np.diff(times) > MAX_ORBIT_STEP_S)
    lower = int(np.searchsorted(times, met, side="right")) - 1
    if lower < 0 or met > times[-1] or (met > times[lower] and lower in breaks):
        raise ValueError("POSHIST не покрывает заданный момент")
    begins = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(times) - 1]
    segment = np.searchsorted(ends, lower)
    start, stop = begins[segment], ends[segment]

    separation, radius, margin = geometry(met)
    result = {
        "earth_occulted": bool(margin < 0),
        "geocenter_separation_deg": float(separation),
        "earth_angular_radius_deg": float(radius),
        "limb_margin_deg": float(margin),
        "in_saa": bool(saa[lower]),
    }
    if margin < 0:
        samples = times[start:stop + 1]
        margins = geometry(samples)[2]
        crossings = np.flatnonzero((margins[:-1] < 0) != (margins[1:] < 0))
        roots = [
            brentq(lambda at: geometry(at)[2], samples[i], samples[i + 1])
            for i in crossings
        ]
        left = next((root for root in reversed(roots) if root <= met), None)
        right = next((root for root in roots if root > met), None)
        result["occultation_interval_utc"] = [
            utc(boundary) if boundary is not None else None for boundary in (left, right)
        ]

    if saa[lower]:
        changes = np.flatnonzero(saa[start:stop] != saa[start + 1:stop + 1]) + start + 1
        before, after = changes[changes <= lower], changes[changes > lower]
        result["saa_interval_utc"] = [
            utc(times[before[-1]]) if before.size else None,
            utc(times[after[0]]) if after.size else None,
        ]

    if window:
        first, last = window
        result["window_utc"] = [utc(first), utc(last)]
        if times[start] <= first <= last <= times[stop]:
            samples = np.r_[first, times[(times > first) & (times < last)], last]
            margins = geometry(samples)[2]
            low, high = float(margins.min()), float(margins.max())
            result["window_limb_margin_range_deg"] = [low, high]
            result["window_visibility"] = (
                "occulted" if high < 0 else "clear" if low >= 0 else "mixed"
            )
        else:
            result["window_visibility"] = "unknown_orbit_gap"
    return result


def ctime_check(path, met):
    with fits.open(path) as hdus:
        header = hdus[0].header
        if not header["TSTART"] <= met < header["TSTOP"]:
            raise ValueError("файл CTIME не покрывает заданный момент")
        gti = hdus["GTI"].data
        bins = hdus["SPECTRUM"].data
        has_gti = bool(np.any((gti["START"] <= met) & (met < gti["STOP"])))
        selected = bins[(bins["TIME"] <= met) & (met < bins["ENDTIME"])]
        usable = bool(np.any((selected["QUALITY"] == 0) & (selected["EXPOSURE"] > 0)))
        result = {
            "in_gti": has_gti,
            "usable_bin": has_gti and usable,
            "bins": [
                {"exposure_s": float(row["EXPOSURE"]), "quality": int(row["QUALITY"])}
                for row in selected
            ],
        }
        if not has_gti:
            previous = gti["STOP"][gti["STOP"] <= met]
            following = gti["START"][gti["START"] > met]
            result["gti_gap_utc"] = [
                utc(np.max(previous)) if previous.size else None,
                utc(np.min(following)) if following.size else None,
            ]
        return result


def catalog_check(met, bounds, source):
    lo, hi = [(EPOCH + TimeDelta(at, format="sec")).utc.mjd for at in bounds]
    query = (
        "SELECT trigger_name, name, trigger_time, trigger_type, ra, dec, error_radius "
        f"FROM fermigtrig WHERE trigger_time BETWEEN {lo:.12f} AND {hi:.12f} "
        "ORDER BY trigger_time"
    )
    url = TAP + "?" + urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": query})
    with urlopen(url, timeout=30) as response:
        document = votable.parse(io.BytesIO(response.read()))
    resource = document.resources[0]
    if any(info.name == "QUERY_STATUS" and info.value != "OK" for info in resource.infos):
        raise ValueError("HEASARC не вернул полную выборку каталога")
    rows = document.get_first_table().to_table()

    def number(row, key):
        value = row[key]
        return float(value) if not np.ma.is_masked(value) and np.isfinite(value) else None

    candidates = []
    for row in rows:
        trigger = Time(row["trigger_time"], format="mjd", scale="utc")
        ra, dec = number(row, "ra"), number(row, "dec")
        separation = None
        if ra is not None and dec is not None and 0 <= ra <= 360 and -90 <= dec <= 90:
            separation = float(source.separation(SkyCoord(ra, dec, unit="deg")).deg)
        candidates.append({
            "id": str(row["trigger_name"]),
            "name": str(row["name"]),
            "type": str(row["trigger_type"]),
            "utc": trigger.isot + "Z",
            "offset_s": round((trigger - EPOCH).sec - met, 3),
            "separation_deg": separation,
            "statistical_error_radius_deg": number(row, "error_radius"),
        })
    return {"search_utc": [utc(at) for at in bounds], "candidates": candidates, "source": url}


def check(args, t0):
    met = (t0 - EPOCH).sec
    day = date.fromisoformat(t0.isot[:10])
    source = SkyCoord(args.ra, args.dec, unit="deg", frame="icrs")
    window = tuple(met + offset for offset in args.window) if args.window else None
    result = {"utc": utc(met), "fermi_met": met, "ra_deg": args.ra, "dec_deg": args.dec}
    issues, orbit_sources, orbits = {}, [], []

    def product(for_day, name):
        return archive_file(for_day, name, args.cache_dir, args.data_dir, args.offline)

    # One orbit on either side is enough to bracket a normal occultation.
    scan_begin = min(met - 6000, window[0]) if window else met - 6000
    scan_end = max(met + 6000, window[1]) if window else met + 6000
    first_day, last_day = [date.fromisoformat(utc(at)[:10]) for at in (scan_begin, scan_end)]
    days = [first_day + timedelta(days=i) for i in range((last_day - first_day).days + 1)]
    for orbit_day in [day, *(item for item in days if item != day)]:
        try:
            path, url = product(orbit_day, "poshist_all")
            orbits.append(read_orbit(path))
            orbit_sources.append(url)
        except (OSError, ValueError, KeyError, TypeError, AstropyUserWarning) as exc:
            issues[f"poshist_{orbit_day}"] = str(exc)

    result["orbit"] = None
    if orbits:
        try:
            result["orbit"] = orbit_check(met, window, source.cartesian.xyz.value, orbits)
            result["orbit"]["sources"] = orbit_sources
        except ValueError as exc:
            issues["orbit"] = str(exc)

    result["ctime"] = None
    try:
        path, url = product(day, f"ctime_{args.detector}")
        result["ctime"] = ctime_check(path, met)
        result["ctime"].update(detector=args.detector, source=url)
    except (OSError, ValueError, KeyError, TypeError, AstropyUserWarning) as exc:
        issues["ctime"] = str(exc)

    result["catalog"] = None
    if not args.offline:
        bounds = (met - args.trigger_seconds, met + args.trigger_seconds)
        if window:
            bounds = (min(bounds[0], window[0]), max(bounds[1], window[1]))
        try:
            result["catalog"] = catalog_check(met, bounds, source)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            issues["catalog"] = str(exc)
    result["issues"] = issues
    return result


def print_report(result):
    print(f'{result["utc"]}  RA={result["ra_deg"]}°  Dec={result["dec_deg"]}°  (J2000)')
    orbit = result["orbit"]
    ctime = result["ctime"]
    catalog = result["catalog"]
    if orbit:
        print("Земля закрывала источник." if orbit["earth_occulted"] else "Источник был над краем Земли.")
        print(f'Угол до центра Земли {orbit["geocenter_separation_deg"]:.2f}°, '
              f'радиус диска {orbit["earth_angular_radius_deg"]:.2f}°, '
              f'расстояние от края со знаком {orbit["limb_margin_deg"]:+.2f}°.')
        if "occultation_interval_utc" in orbit:
            start, stop = (bound or "вне доступной орбиты" for bound in orbit["occultation_interval_utc"])
            print(f"Затенение (приблизительно) — {start} … {stop}")
        if "window_visibility" in orbit:
            labels = {
                "occulted": "закрыт на всем интервале", "clear": "виден на всем интервале",
                "mixed": "видимость менялась", "unknown_orbit_gap": "орбитальных данных недостаточно",
            }
            print(f'Заданное окно — {labels[orbit["window_visibility"]]}.')
        print("SAA — внутри." if orbit["in_saa"] else "SAA — вне аномалии.")
        if "saa_interval_utc" in orbit:
            start, stop = (bound or "вне доступной орбиты" for bound in orbit["saa_interval_utc"])
            print(f"Проход SAA — {start} … {stop}")
    else:
        print("Геометрия и SAA не определены.")
    if ctime:
        status = "есть пригодный бин" if ctime["usable_bin"] else "пригодного бина нет"
        print(f'CTIME {ctime["detector"]} — {status}.')
        if "gti_gap_utc" in ctime:
            start, stop = (bound or "вне файла CTIME" for bound in ctime["gti_gap_utc"])
            print(f"Пропуск между интервалами пригодных данных — {start} … {stop}")
    else:
        print("CTIME — данных для проверки нет.")
    if catalog is None:
        print("FERMIGTRIG — каталог не проверен.")
    else:
        print(f'Окно поиска в FERMIGTRIG — {" … ".join(catalog["search_utc"])}')
        if not catalog["candidates"]:
            print("В этом окне каталог не содержит бортовых триггеров.")
        for row in catalog["candidates"]:
            angle = row["separation_deg"]
            radius = row["statistical_error_radius_deg"]
            position = f", отступ {angle:.2f}°" if angle is not None else ", локализация недоступна"
            error = f", статистическая ошибка {radius:.2f}°" if radius is not None else ""
            print(f'  {row["id"]} {row["name"]} ({row["type"]}), '
                  f'T₀{row["offset_s"]:+.3f} с{position}{error}')
    if orbit and (orbit["earth_occulted"] or orbit["in_saa"]):
        print("Прямое наблюдение источника в заданный момент было невозможно.")
    elif catalog and catalog["candidates"]:
        print("Связь кандидатов с источником требует проверки.")
    print("Поиск слабого сигнала в непрерывных данных не выполнялся.")
    for part, issue in result["issues"].items():
        print(f"{part} — {issue}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("time", help="UTC in ISO format, e.g. 2026-08-08T14:20:58Z")
    parser.add_argument("ra", type=float, help="J2000 RA in degrees")
    parser.add_argument("dec", type=float, help="J2000 Dec in degrees")
    parser.add_argument("--window", nargs=2, type=float, metavar=("START", "END"),
                        help="visibility interval in seconds relative to the input time")
    parser.add_argument("--detector", default="n0", choices=[*("n" + x for x in "0123456789ab"), "b0", "b1"])
    parser.add_argument("--trigger-seconds", type=float, default=60, help="catalog half-window in seconds (default: 60)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-dir", type=Path, action="append", default=[],
                        help="local directory with archive FITS files (repeatable)")
    parser.add_argument("--offline", action="store_true", help="use local FITS only and skip the live catalog")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the text summary")
    args = parser.parse_args(argv)
    if not (0 <= args.ra < 360 and -90 <= args.dec <= 90):
        parser.error("RA must be in [0, 360), Dec in [-90, 90] degrees")
    if not math.isfinite(args.trigger_seconds) or args.trigger_seconds <= 0:
        parser.error("--trigger-seconds must be finite and positive")
    if args.window and (not all(map(math.isfinite, args.window)) or args.window[0] > args.window[1]):
        parser.error("--window requires two finite, ordered offsets")
    try:
        t0 = Time(args.time, format="isot", scale="utc")
    except ValueError:
        parser.error("time must be ISO UTC, e.g. 2026-08-08T14:20:58Z")
    if t0 < Time("2008-06-11", scale="utc"):
        parser.error("time precedes the Fermi launch")
    result = check(args, t0)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result)
    complete = result["orbit"] is not None and result["ctime"] is not None
    if args.window and result["orbit"]:
        complete &= result["orbit"]["window_visibility"] != "unknown_orbit_gap"
    return 0 if complete and (args.offline or result["catalog"] is not None) else 1


if __name__ == "__main__":
    sys.exit(main())
