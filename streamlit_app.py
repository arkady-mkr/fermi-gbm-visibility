"""Interactive visibility check against the public NASA HEASARC archive."""

from datetime import date
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from types import SimpleNamespace
import logging
import re

import streamlit as st
from astropy.time import Time

from gbm_visibility import check

CACHE = Path(gettempdir()) / "fermi-gbm-streamlit"
UTC_FORMAT = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ](?:[01]\d|2[0-3]):[0-5]\d:(?:[0-5]\d|60)(?:\.\d{1,9})?Z?"
)
EXAMPLE = {"utc": "2026-08-08T14:20:58Z", "ra": "345.23359", "dec": "8.47848"}
LOGGER = logging.getLogger(__name__)


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


def render_result(result):
    orbit, ctime, catalog = (result[key] for key in ("orbit", "ctime", "catalog"))
    st.header("Результат")
    st.caption(f'{result["utc"]} / RA {result["ra_deg"]:.8g}° / Dec {result["dec_deg"]:.8g}° (J2000)')

    st.subheader("Затенение Землей")
    if orbit:
        if orbit["earth_occulted"]:
            st.warning("В заданный момент Земля закрывала источник.")
            st.text("Интервал затенения (приблизительно), UTC")
            st.text(interval(orbit["occultation_interval_utc"]))
        else:
            st.success("В заданный момент источник был над краем Земли.")
        st.write(
            f'Угол до центра Земли **{orbit["geocenter_separation_deg"]:.2f}°**. '
            f'Радиус земного диска **{orbit["earth_angular_radius_deg"]:.2f}°**. '
            f'Отступ от края **{orbit["limb_margin_deg"]:+.2f}°**.'
        )
    else:
        st.warning("Орбитальных данных для расчета геометрии нет.")

    st.subheader("SAA и данные CTIME n0")
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
        st.warning("Данные CTIME n0 для проверки недоступны.")

    st.subheader("Бортовые триггеры GBM")
    if catalog is None:
        st.warning("Каталог FERMIGTRIG недоступен.")
    elif catalog["candidates"]:
        st.write("Триггеры в пределах ±60 секунд. Связь с источником требует проверки.")
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

    with st.expander("Исходные данные"):
        for number, url in enumerate(orbit["sources"] if orbit else [], 1):
            st.link_button(f"NASA HEASARC POSHIST {number}", url)
        if ctime:
            st.link_button("NASA HEASARC CTIME n0", ctime["source"])
        if catalog:
            st.link_button("Запрос к каталогу FERMIGTRIG", catalog["source"])
        if result["issues"]:
            st.write("Некоторые продукты архива не удалось обработать.")
            st.text("\n".join(result["issues"]))


def fill_example():
    st.session_state.update(EXAMPLE)
    st.session_state.pop("result", None)


st.set_page_config(page_title="Видимость источника для Fermi GBM", layout="centered")
st.title("Видимость источника для Fermi GBM")
st.write("Проверьте, закрывала ли Земля источник в заданный момент.")
st.button("Заполнить пример GRB 260808B", on_click=fill_example)

with st.form("visibility"):
    utc = st.text_input("Время UTC", key="utc", placeholder="2026-08-08T14:20:58Z", max_chars=40)
    ra_col, dec_col = st.columns(2)
    ra = ra_col.text_input("RA (J2000, °)", key="ra", placeholder="345.23359", max_chars=24)
    dec = dec_col.text_input("Dec (J2000, °)", key="dec", placeholder="8.47848", max_chars=24)
    submitted = st.form_submit_button("Проверить", type="primary")

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

st.caption("Модель сферической Земли с радиусом 6371 км. Для поиска слабого сигнала нужен отдельный анализ.")
