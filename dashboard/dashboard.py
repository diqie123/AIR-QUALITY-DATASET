from __future__ import annotations

import os

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")

import io
import re
import zipfile
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener
import pandas as pd
import streamlit as st

DATASET_NAME = "Air Quality Dataset"
GOOGLE_DRIVE_VIEW_URL = "https://drive.google.com/file/d/1RhU3gJlkteaAQfyn9XOVAz7a5o1-etgr/view"


def _extract_google_drive_file_id(view_url: str) -> str:
    match = re.search(r"/d/([A-Za-z0-9_-]{10,})", view_url)
    if match is not None:
        return match.group(1)

    match = re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", view_url)
    if match is not None:
        return match.group(1)

    raise ValueError("Link Google Drive tidak valid. Pastikan formatnya seperti .../file/d/<FILE_ID>/view.")


GOOGLE_DRIVE_FILE_ID = _extract_google_drive_file_id(GOOGLE_DRIVE_VIEW_URL)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _download_google_drive_file(file_id: str) -> bytes:
    jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))

    def _fetch(url: str) -> tuple[str, bytes]:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/octet-stream,*/*",
            },
        )
        with opener.open(req) as resp:
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read()
        return content_type, body

    base_url = "https://drive.google.com/uc"
    alt_url = "https://drive.usercontent.google.com/download"

    for direct_base in (alt_url, base_url):
        content_type, body = _fetch(
            f"{direct_base}?{urlencode({'export': 'download', 'confirm': 't', 'id': file_id})}"
        )
        if "text/html" not in content_type.lower():
            return body
        if body[:4] == b"PK\x03\x04":
            return body

    content_type, body = _fetch(f"{base_url}?{urlencode({'export': 'download', 'id': file_id})}")
    if "text/html" not in content_type.lower():
        return body
    if body[:4] == b"PK\x03\x04":
        return body

    confirm_token = None
    for cookie in jar:
        if cookie.name.startswith("download_warning"):
            confirm_token = cookie.value
            break

    text = body.decode("utf-8", errors="ignore")
    if confirm_token is None:
        m = re.search(r"confirm=([0-9A-Za-z_-]+)", text)
        if m is not None:
            confirm_token = m.group(1)
    if confirm_token is None:
        m = re.search(r'name="confirm"\s+value="([^"]+)"', text)
        if m is not None:
            confirm_token = m.group(1)

    if confirm_token is None:
        confirm_token = "t"

    for confirm_base in (alt_url, base_url):
        content_type2, body2 = _fetch(
            f"{confirm_base}?{urlencode({'export': 'download', 'confirm': confirm_token, 'id': file_id})}"
        )
        if "text/html" not in content_type2.lower():
            return body2
        if body2[:4] == b"PK\x03\x04":
            return body2

    raise RuntimeError(
        "Gagal mengunduh dataset dari Google Drive. Unduh manual dari tautan sumber, lalu ekstrak file CSV "
        "ke folder submission/data (mis. PRSA_Data_*.csv)."
    )


def _extract_zip_bytes(zip_bytes: bytes, out_dir: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            if not member.lower().endswith(".csv"):
                continue
            target_name = Path(member).name
            out_path = out_dir / target_name
            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())


@st.cache_data(show_spinner=False)
def _ensure_air_quality_files() -> list[Path]:
    data_dir = _project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    def _canon(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    required_any = {"pm25", "pm10", "so2", "no2", "co", "o3"}

    def _looks_like_air_quality_csv(p: Path) -> bool:
        try:
            head = pd.read_csv(p, nrows=5)
        except Exception:
            return False
        cols = {_canon(c) for c in head.columns.astype(str).tolist()}
        has_pollutant = any(x in cols for x in required_any) or any("pm25" in x for x in cols)
        has_time = any(x in cols for x in {"datetime", "timestamp", "date"}) or {
            "year",
            "month",
            "day",
            "hour",
        }.issubset(cols)
        return has_pollutant and has_time

    existing_files = [p for p in data_dir.glob("*.csv") if p.is_file() and _looks_like_air_quality_csv(p)]
    if existing_files:
        return sorted(existing_files)

    zip_or_csv_bytes = _download_google_drive_file(GOOGLE_DRIVE_FILE_ID)
    if zip_or_csv_bytes[:4] == b"PK\x03\x04":
        _extract_zip_bytes(zip_or_csv_bytes, data_dir)
    else:
        (data_dir / "air_quality_raw.csv").write_bytes(zip_or_csv_bytes)

    extracted_files = [p for p in data_dir.glob("*.csv") if p.is_file() and _looks_like_air_quality_csv(p)]
    if extracted_files:
        return sorted(extracted_files)

    raise FileNotFoundError(
        "Dataset Air Quality belum siap. Pastikan file CSV dataset Air Quality ada di folder submission/data "
        "atau tautan Google Drive dapat diakses."
    )


def _canonical_col_name(col: str) -> str:
    c = col.strip().lower()
    c = c.replace(" ", "_")
    c = c.replace("-", "_")
    return c


def _safe_chart_key(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(label))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "value"


def _clean_measurement_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.mask(values == 999)
    values = values.mask(values < 0)
    return values


def _detect_station_column(df: pd.DataFrame) -> str | None:
    candidates = ["station", "site", "location", "station_id", "stationname"]
    canon = {_canonical_col_name(c): c for c in df.columns.astype(str).tolist()}
    for want in candidates:
        if want in canon:
            return canon[want]
    return None


def _build_datetime(df: pd.DataFrame) -> pd.Series:
    canon_map = {_canonical_col_name(c): c for c in df.columns.astype(str).tolist()}

    if "datetime" in canon_map:
        return pd.to_datetime(df[canon_map["datetime"]], errors="coerce", utc=False)
    if "timestamp" in canon_map:
        return pd.to_datetime(df[canon_map["timestamp"]], errors="coerce", utc=False)

    if "date" in canon_map and "time" in canon_map:
        return pd.to_datetime(
            df[canon_map["date"]].astype(str).str.strip() + " " + df[canon_map["time"]].astype(str).str.strip(),
            errors="coerce",
            utc=False,
        )

    required_parts = ["year", "month", "day", "hour"]
    if all(p in canon_map for p in required_parts):
        parts = df[[canon_map[p] for p in required_parts]].copy()
        parts.columns = required_parts
        for p in required_parts:
            parts[p] = pd.to_numeric(parts[p], errors="coerce")
        parts = parts.dropna()
        return pd.to_datetime(
            dict(
                year=parts["year"].astype(int),
                month=parts["month"].astype(int),
                day=parts["day"].astype(int),
                hour=parts["hour"].astype(int),
            ),
            errors="coerce",
            utc=False,
        )

    return pd.to_datetime(pd.Series([pd.NaT] * len(df)), errors="coerce", utc=False)


def _pollutant_columns(df: pd.DataFrame) -> dict[str, str]:
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    result: dict[str, str] = {}
    for col in df.columns.astype(str).tolist():
        n = norm(col)
        if n in {"pm10", "so2", "no2", "co", "o3"}:
            result[n.upper()] = col
        elif n in {"pm25", "pm2p5", "pm25ugm3"} or "pm25" in n:
            result["PM2.5"] = col
    preferred_order = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    ordered: dict[str, str] = {}
    for k in preferred_order:
        if k in result:
            ordered[k] = result[k]
    for k, v in result.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


@st.cache_data(show_spinner=False)
def _load_air_quality_data() -> pd.DataFrame:
    csv_files = _ensure_air_quality_files()
    frames: list[pd.DataFrame] = []
    for p in csv_files:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError("Tidak bisa membaca file CSV Air Quality dari folder data.")

    df = pd.concat(frames, ignore_index=True)
    df = df.copy()

    dt = _build_datetime(df)
    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime")
    df["date"] = df["datetime"].dt.date
    df["year_month"] = df["datetime"].dt.to_period("M").astype(str)
    df["hour"] = df["datetime"].dt.hour
    df["day_name"] = df["datetime"].dt.day_name()

    return df


def main() -> None:
    st.set_page_config(page_title=f"{DATASET_NAME} Dashboard", layout="wide")
    st.title(f"{DATASET_NAME} Dashboard")

    with st.expander("Sumber Dataset", expanded=False):
        st.write(GOOGLE_DRIVE_VIEW_URL)

    try:
        df = _load_air_quality_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    pollutant_map = _pollutant_columns(df)
    if not pollutant_map:
        st.error("Kolom polutan tidak ditemukan pada dataset.")
        st.stop()

    station_col = _detect_station_column(df)

    with st.sidebar:
        st.header("Filter")

        min_date = pd.to_datetime(df["date"]).min().date()
        max_date = pd.to_datetime(df["date"]).max().date()
        start_date, end_date = st.date_input(
            "Rentang tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        selected_station: str | None = None
        if station_col is not None:
            stations = sorted(df[station_col].dropna().astype(str).unique().tolist())
            if stations:
                selected_station = st.selectbox("Station", options=["(Semua)"] + stations, index=0)

        pollutant_label = st.selectbox(
            "Polutan",
            options=list(pollutant_map.keys()),
            index=0 if "PM2.5" not in pollutant_map else list(pollutant_map.keys()).index("PM2.5"),
        )
        freq_label = st.selectbox("Agregasi tren", options=["Harian", "Mingguan", "Bulanan"], index=0)

    filtered_df = df.copy()

    start_d = pd.to_datetime(start_date).date()
    end_d = pd.to_datetime(end_date).date()
    dates = pd.to_datetime(filtered_df["date"], errors="coerce").dt.date
    filtered_df = filtered_df[(dates >= start_d) & (dates <= end_d)].copy()

    if station_col is not None and selected_station is not None and selected_station != "(Semua)":
        filtered_df = filtered_df[filtered_df[station_col].astype(str) == str(selected_station)].copy()

    if filtered_df.empty:
        st.warning("Data kosong untuk filter yang dipilih. Ubah filter di sidebar.")
        return

    pollutant_col = pollutant_map[pollutant_label]
    safe_pollutant_key = _safe_chart_key(pollutant_label)
    clean_vals = _clean_measurement_series(filtered_df[pollutant_col])
    avg_val = float(clean_vals.mean()) if not clean_vals.dropna().empty else None
    max_val = float(clean_vals.max()) if not clean_vals.dropna().empty else None
    missing_rate = float(clean_vals.isna().mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Jumlah baris", f"{len(filtered_df):,}")
    if station_col is None:
        c2.metric("Jumlah station", "-")
    else:
        c2.metric("Jumlah station", f"{filtered_df[station_col].nunique():,}")
    c3.metric(f"Rata-rata {pollutant_label}", "-" if avg_val is None else f"{avg_val:,.2f}")
    c4.metric(f"Maksimum {pollutant_label}", "-" if max_val is None else f"{max_val:,.2f}")
    c5.metric("Missing rate", f"{missing_rate*100:.1f}%")

    st.subheader(f"Tren {pollutant_label} dari Waktu ke Waktu")
    rule = {"Harian": "D", "Mingguan": "W", "Bulanan": "M"}[freq_label]
    series_df = filtered_df[["datetime"]].copy()
    series_df["_val"] = clean_vals
    series_df = series_df.dropna(subset=["_val"]).copy()
    ts = series_df.set_index("datetime")["_val"].resample(rule).mean().dropna()
    if ts.empty:
        st.info("Tidak ada data valid untuk tren pada rentang filter ini.")
    else:
        st.line_chart(ts.rename(safe_pollutant_key).to_frame())

    st.subheader(f"Pola {pollutant_label} Berdasarkan Jam")
    by_hour = (
        filtered_df.assign(_val=clean_vals)
        .dropna(subset=["_val"])
        .groupby("hour", as_index=True)["_val"]
        .mean()
        .sort_index()
    )
    if by_hour.empty:
        st.info("Tidak ada data valid untuk agregasi jam pada rentang filter ini.")
    else:
        hour_df = (
            pd.DataFrame({"hour": list(range(24))})
            .merge(by_hour.rename(safe_pollutant_key).reset_index(), on="hour", how="left")
            .sort_values("hour")
        )
        try:
            import altair as alt  # type: ignore

            st.altair_chart(
                alt.Chart(hour_df)
                .mark_bar()
                .encode(
                    x=alt.X("hour:O", axis=alt.Axis(title="Jam", labelAngle=0)),
                    y=alt.Y(f"{safe_pollutant_key}:Q", axis=alt.Axis(title=f"Rata-rata {pollutant_label}")),
                ),
                use_container_width=True,
            )
        except Exception:
            st.bar_chart(hour_df.set_index("hour")[[safe_pollutant_key]])

    st.subheader(f"Top Station Berdasarkan Rata-rata {pollutant_label}")
    if station_col is None:
        st.info("Kolom station tidak tersedia pada dataset.")
    else:
        station_rank = (
            filtered_df.assign(_val=clean_vals)
            .dropna(subset=["_val"])
            .groupby(station_col, as_index=False)["_val"]
            .mean()
            .sort_values("_val", ascending=False)
            .head(10)
            .rename(columns={"_val": f"avg_{safe_pollutant_key}"})
        )
        if station_rank.empty:
            st.info("Tidak ada data station yang valid pada rentang filter ini.")
        else:
            plot_df = station_rank.copy()
            try:
                import altair as alt  # type: ignore

                st.altair_chart(
                    alt.Chart(plot_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(station_col, sort="-y", axis=alt.Axis(title="Station", labelAngle=-45)),
                        y=alt.Y(f"avg_{safe_pollutant_key}:Q", axis=alt.Axis(title=f"Rata-rata {pollutant_label}")),
                    ),
                    use_container_width=True,
                )
            except Exception:
                st.bar_chart(plot_df.set_index(station_col)[[f"avg_{safe_pollutant_key}"]])
            st.dataframe(station_rank, hide_index=True, use_container_width=True)

    st.subheader("Data (Preview)")
    st.dataframe(filtered_df.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()
