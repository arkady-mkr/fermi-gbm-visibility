"""Interactive visibility check against the public NASA HEASARC archive."""

from datetime import date
from math import hypot
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from types import SimpleNamespace
import logging
import re

import streamlit as st
from astropy.coordinates import SkyCoord
from astropy.time import Time

from gbm_visibility import check

CACHE = Path(gettempdir()) / "fermi-gbm-streamlit"
UTC_FORMAT = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ](?:[01]\d|2[0-3]):[0-5]\d:(?:[0-5]\d|60)(?:\.\d{1,9})?Z?"
)
LOGGER = logging.getLogger(__name__)
TITLE = "Видимость источника для Fermi GBM"

# UTC, RA/Dec J2000, published GBM outcome, parameter and outcome references.
EXAMPLES = {
    "GRB 260808B": ("2026-08-08T14:20:58Z", "345.23359", "8.47848", None, (45316, 45318)),
    "GRB 231111A": ("2023-11-11T14:17:19Z", "290.197", "52.440", None, (34981, 35033)),
    "GRB 260808A": ("2026-08-08T02:51:42.310Z", "341.8", "14.1", "trigger", (45313,)),
    "GRB 170817A": ("2017-08-17T12:41:06.47Z", "197.450371", "-23.381486", "trigger", (21529, 21520)),
    "GRB 251205A": ("2025-12-05T23:39:47Z", "200.27223", "29.92888", "targeted", (43005, 43059)),
    "GRB 211203A": ("2021-12-03T09:02:38Z", "170.174", "16.256", "upper_limit", (31160, 31163)),
}

FORM_CSS = """
<style>
  .st-key-gbm-heading h1 { font-size: clamp(1.75rem, 4vw, 2.25rem); }
  @media (min-width: 800px) { .st-key-gbm-heading h1 { white-space: nowrap; } }
  [data-testid="stTextInput"] input::placeholder { color: currentColor; opacity: 0.62; }
</style>
"""


@st.cache_resource
def calculation_lock():
    return Lock()


def parse_event(utc, ra, dec):
    utc = utc.strip().replace(" ", "T")
    if not UTC_FORMAT.fullmatch(utc):
        raise ValueError("Введите время UTC в формате ГГГГ-ММ-ДДTчч:мм:сс.")
    day = date.fromisoformat(utc[:10])
    if day < date(2008, 6, 11) or day > Time.now().utc.datetime.date():
        raise ValueError("Выберите дату от запуска Fermi до сегодняшнего дня.")
    instant = Time(utc, format="isot", scale="utc", precision=9)
    if utc[17:19] == "60" and instant.utc.isot[17:19] != "60":
        raise ValueError("Секунда 60 допустима только для секунды координации.")
    try:
        ra, dec = (float(value.strip().replace(",", ".")) for value in (ra, dec))
    except ValueError:
        raise ValueError("Введите RA и Dec числами в градусах.") from None
    if not (0 <= ra < 360 and -90 <= dec <= 90):
        raise ValueError("RA должно быть от 0 до 360° без 360°, Dec – от −90 до +90°.")
    return instant, ra, dec


def calculate(instant, ra, dec):
    args = SimpleNamespace(
        ra=ra, dec=dec, window=None, detector="n0", trigger_seconds=60,
        offline=False, data_dir=[], cache_dir=CACHE,
    )
    with calculation_lock():
        try:
            return check(args, instant)
        finally:
            try:
                products = [
                    path for path in CACHE.glob("glg_*")
                    if path.is_file() and path.name.endswith((".fit", ".pha", ".fit.gz", ".pha.gz"))
                ]
                for path in sorted(products, key=lambda path: path.stat().st_mtime)[:-12]:
                    path.unlink()
            except OSError:
                LOGGER.warning("Could not trim FITS cache", exc_info=True)


def interval(bounds):
    return " – ".join(value or "граница не определена" for value in bounds)


def example_for(result):
    instant = Time(result["utc"], format="isot", scale="utc")
    position = SkyCoord(result["ra_deg"], result["dec_deg"], unit="deg")
    for name, (utc, ra, dec, outcome, references) in EXAMPLES.items():
        if abs((instant - Time(utc, format="isot", scale="utc")).sec) > 1:
            continue
        if position.separation(SkyCoord(float(ra), float(dec), unit="deg")).deg <= 0.1:
            return name, outcome, references
    return None, None, ()


def signal_answer(orbit, catalog, published_outcome):
    if published_outcome == "trigger":
        return "success", "Да. Команда Fermi GBM сообщила об обнаружении и бортовом триггере."
    if published_outcome == "targeted":
        return "success", "Да. Команда Fermi GBM обнаружила сигнал наземным целевым поиском."
    if published_outcome == "upper_limit":
        return "info", "Значимый сигнал не обнаружен. Команда Fermi GBM опубликовала верхние пределы."
    if orbit and orbit["earth_occulted"]:
        return "info", "Прямой сигнал от источника в этот момент был недоступен из-за затенения Землей."
    if orbit and orbit["in_saa"]:
        return "warning", "В этот момент Fermi находился в SAA. Наблюдение источника недоступно."
    if catalog is None:
        return "warning", "Не удалось проверить. Каталог бортовых триггеров недоступен."
    for row in catalog["candidates"]:
        separation = row["separation_deg"]
        statistical_error = row["statistical_error_radius_deg"]
        if (
            row["type"].upper() == "GRB"
            and abs(row["offset_s"]) <= 10
            and separation is not None
            and statistical_error is not None
            and separation <= hypot(statistical_error, 3.7)
        ):
            return "info", f'Найден совместимый бортовой триггер {row["id"]}. Связь с источником требует проверки.'
    return "info", "Подтверждения нет. В каталоге не найден совместимый бортовой триггер."


def render_result(result):
    orbit, ctime, catalog = (result[key] for key in ("orbit", "ctime", "catalog"))
    name, published_outcome, references = example_for(result)
    st.header(name or "Результат")
    st.caption(f'`{result["utc"]}` / RA {result["ra_deg"]:.8g}° / Dec {result["dec_deg"]:.8g}° (J2000)')

    st.subheader("Было ли затенение Землей?")
    if orbit:
        if orbit["earth_occulted"]:
            st.warning("Да. В заданный момент Земля закрывала источник.")
            st.text("Интервал затенения (приблизительно), UTC")
            st.text(interval(orbit["occultation_interval_utc"]))
        else:
            st.success("Нет. В заданный момент источник был над краем Земли.")
        st.write(
            f'Угол до центра Земли **{orbit["geocenter_separation_deg"]:.2f}°**. '
            f'Радиус земного диска **{orbit["earth_angular_radius_deg"]:.2f}°**. '
            f'Отступ от края **{orbit["limb_margin_deg"]:+.2f}°**.'
        )
    else:
        st.warning("Орбитальных данных для расчета геометрии нет.")

    st.subheader("Есть ли сигнал в данных GBM?")
    status, message = signal_answer(orbit, catalog, published_outcome)
    {"info": st.info, "warning": st.warning, "success": st.success}[status](message)
    if not published_outcome and not (orbit and (orbit["earth_occulted"] or orbit["in_saa"])):
        st.caption("Для нового события форма проверяет каталог триггеров. Поиск слабого сигнала в непрерывных данных требует отдельного анализа.")

    with st.expander("Данные наблюдения и источники"):
        if orbit:
            st.write(
                "Fermi находился в Южно-Атлантической аномалии (SAA)."
                if orbit["in_saa"] else "Fermi находился вне Южно-Атлантической аномалии (SAA)."
            )
            if "saa_interval_utc" in orbit:
                st.text(interval(orbit["saa_interval_utc"]))
        if ctime:
            st.write(
                "В CTIME n0 есть пригодный бин."
                if ctime["usable_bin"] else "В CTIME n0 нет пригодного бина."
            )
            if "gti_gap_utc" in ctime:
                st.text("Пропуск между интервалами пригодных данных, UTC")
                st.text(interval(ctime["gti_gap_utc"]))
        else:
            st.write("Данные CTIME n0 для проверки недоступны.")

        if catalog is None:
            st.write("Каталог FERMIGTRIG недоступен.")
        elif catalog["candidates"]:
            st.write("Триггеры в пределах ±60 секунд.")
            st.table([
                {
                    "Триггер": row["id"], "Тип": row["type"], "UTC": row["utc"],
                    "Отступ по времени, с": round(row["offset_s"], 3),
                    "Угловой отступ, °": (
                        round(row["separation_deg"], 2) if row["separation_deg"] is not None else None
                    ),
                }
                for row in catalog["candidates"]
            ])
        else:
            st.write("В каталоге FERMIGTRIG нет бортовых триггеров в пределах ±60 секунд.")

        for number in references:
            st.link_button(f"GCN {number}", f"https://gcn.nasa.gov/circulars/{number}")
        for number, url in enumerate(orbit["sources"] if orbit else [], 1):
            st.link_button(f"NASA HEASARC POSHIST {number}", url)
        if ctime:
            st.link_button("NASA HEASARC CTIME n0", ctime["source"])
        if catalog:
            st.link_button("Запрос к каталогу FERMIGTRIG", catalog["source"])
        if result["issues"]:
            st.write("Некоторые продукты архива не удалось обработать.")
            st.text("\n".join(result["issues"]))


def fill_example(name):
    utc, ra, dec, _, _ = EXAMPLES[name]
    st.session_state.update(utc=utc, ra=ra, dec=dec)
    st.session_state.pop("result", None)


def clear_form():
    st.session_state.update(utc="", ra="", dec="")
    st.session_state.pop("result", None)


st.set_page_config(page_title=TITLE, layout="centered")
st.html(FORM_CSS)
with st.container(key="gbm-heading"):
    st.title(TITLE)
st.write("Примеры всплесков для заполнения формы")
with st.container(horizontal=True, gap="small"):
    for name in EXAMPLES:
        st.button(name, on_click=fill_example, args=(name,))

with st.form("visibility"):
    utc = st.text_input("Время UTC", key="utc", placeholder="2026-08-08T14:20:58Z", max_chars=40)
    ra_col, dec_col = st.columns(2)
    ra = ra_col.text_input("RA (J2000, °)", key="ra", placeholder="345.23359", max_chars=24)
    dec = dec_col.text_input("Dec (J2000, °)", key="dec", placeholder="8.47848", max_chars=24)
    with st.container(horizontal=True, gap="small"):
        submitted = st.form_submit_button("Проверить", type="primary")
        st.form_submit_button("Очистить", on_click=clear_form)

if submitted:
    st.session_state.pop("result", None)
    try:
        event = parse_event(utc, ra, dec)
    except ValueError as exc:
        st.error(str(exc))
    else:
        with st.spinner("Загружаю данные NASA HEASARC и выполняю расчет…"):
            st.session_state.result = calculate(*event)

if "result" in st.session_state:
    render_result(st.session_state.result)
