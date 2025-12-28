#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
windy_safi_helmholtz_openmeteo.py
---------------------------------
Fusion station GP2 + champ Open-Meteo via équation de Helmholtz 2D
(Dirichlet : φ = 0 sur les bords de la grille).

Pipeline :
  1) Placer GP2 sur la grille Safi
  2) Lire la dernière mesure de GP2 (fichiers GP2_*.txt)
  3) Récupérer les données "current" Open-Meteo sur les 4 coins du domaine
  4) Construire un champ 2D (u, v, T) par interpolation bilinéaire
  5) Calculer le biais station vs modèle au point GP2
  6) Construire un RHS "Dirac discret" pour chaque variable
  7) Résoudre Helmholtz pour u, v, T
  8) Ajouter la correction φ au champ modèle
  9) Visualiser φ_u et le champ de vent corrigé
 10) Vérifier la cohérence au point station

Attention : ce script fait des appels HTTP à l'API Open-Meteo (nécessite internet).
"""

# ============================================================
# PARTIE 1/10 : Imports, dataclasses & configuration
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict
import re

import numpy as np
import pandas as pd  # pas indispensable ici, mais utile si tu veux loguer
import matplotlib.pyplot as plt
import requests  # pour l'appel API Open-Meteo
import time
import os
import xarray as xr
from datetime import datetime, timedelta

GFS_DIR = "gfs_data"
os.makedirs(GFS_DIR, exist_ok=True)

def download_gfs_forecast():
    """Télécharge le fichier GFS 0.25° le plus récent"""
    now = datetime.utcnow()
    run_hour = now.replace(minute=0, second=0, microsecond=0)

    # Les runs GFS sont disponibles à H=00,06,12,18
    run_hour -= timedelta(hours=run_hour.hour % 6)

    run_str = run_hour.strftime("%Y%m%d%H")

    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
        f"file=gfs.t{run_hour.hour:02d}z.pgrb2.0p25.f000"
        f"&all_lev=on&all_var=on&subregion="
        f"&leftlon=-20&rightlon=20&toplat=50&bottomlat=0"  # région Maroc élargie
        f"&dir=/gfs.{run_hour.strftime('%Y%m%d')}/{run_hour.hour:02d}/"
    )

    local_path = f"{GFS_DIR}/gfs_{run_str}.grib2"

    if not os.path.exists(local_path):
        print(f"Downloading GFS {run_str}...")
        r = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(r.content)

    return local_path


@dataclass
class GridConfig:
    """
    Configuration de la grille de calcul (Helmholtz + champ modèle).
    """
    nx: int = 80
    ny: int = 80
    dx: float = 1.0        # pas spatial fictif pour Helmholtz (même en "indices")
    sigma: float = 6.0     # longueur de corrélation (en unités de dx)

    # Domaine Safi + environs (coords géographiques)
    lat_min: float = 32.08
    lat_max: float = 32.38
    lon_min: float = -9.43
    lon_max: float = -9.07

@dataclass
class StationConfig:
    """
    Métadonnées de la station (nom, emplacement, répertoire de données).
    """
    name: str = "GP2"
    data_dir: Path = Path(__file__).parent / "data"

    # Coordonnées RÉELLES de la station GP2 (en degrés décimaux)
    lat: float = 32.23233
    lon: float = -9.25156

    # Indices (i, j) sur la grille – remplis dynamiquement
    i_index: int = 0
    j_index: int = 0

    # ---- Paramètres robustesse ----
    wind_units: str = "m/s"           # "m/s" ou "km/h"
    wind_dir_is_towards: bool = False # True si la station donne la direction VERS (et non d'origine)

    # Altitudes (pour mise à l'échelle verticale de la vitesse)
    anemometer_height_m: float = 1.0   # hauteur station (1 m d'après "#@1m")
    model_wind_height_m: float = 10.0  # Open-Meteo ~10 m


GRID_CFG = GridConfig(
    nx=80,
    ny=80,
    dx=1.0,
    sigma=6.0,
    lat_min=32.08,
    lat_max=32.38,
    lon_min=-9.43,
    lon_max=-9.07,
)

STATION_CFG = StationConfig()


@dataclass
class StationMeasurement:
    """
    Représente UNE mesure de la station GP2 (1 ligne de données).
    """
    timestamp: datetime
    power: float
    speed_ms: float
    dir_deg: float
    rh: float
    air_temp_c: float


@dataclass
class ModelFields:
    """
    Champs modèle 2D sur la grille :
      u[i,j], v[i,j], temp[i,j], rh[i,j]
    """
    u: np.ndarray
    v: np.ndarray
    temp: np.ndarray
    rh: np.ndarray


@dataclass
class FusedFields:
    """
    Champs corrigés après fusion station + modèle,
    et champs de correction φ pour chaque variable.
    """
    u_corr: np.ndarray
    v_corr: np.ndarray
    temp_corr: np.ndarray
    rh_corr: np.ndarray
    phi_u: np.ndarray
    phi_v: np.ndarray
    phi_T: np.ndarray
    phi_RH: np.ndarray


# ============================================================
# PARTIE 2/10 : Gestion des fichiers station GP2 (regex, tri)
# ============================================================

# Exemple de nom de fichier : "GP2_03-12-24_07-30-33.txt"
FNAME_REGEX = re.compile(r"GP2_(\d{2}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.txt$")


def extract_ts_from_filename(fname: str) -> Optional[datetime]:
    """
    Extrait un datetime à partir d'un nom de fichier GP2_DD-MM-YY_HH-MM-SS.txt.
    Retourne None si le pattern ne matche pas.
    """
    m = FNAME_REGEX.search(fname)
    if not m:
        return None
    date_str, time_str = m.group(1), m.group(2)
    return datetime.strptime(f"{date_str}_{time_str}", "%d-%m-%y_%H-%M-%S")


def find_latest_gp2_file(station_cfg: StationConfig) -> Path:
    """
    Parcourt station_cfg.data_dir et retourne le chemin du fichier GP2_*.txt
    ayant le timestamp le plus récent dans son nom.
    """
    data_dir = station_cfg.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Dossier de données inexistant : {data_dir}")

    best_path: Optional[Path] = None
    best_ts: Optional[datetime] = None

    for f in data_dir.glob(f"{station_cfg.name}_*.txt"):
        if not f.is_file():
            continue
        ts = extract_ts_from_filename(f.name)
        if ts is None:
            continue
        if best_ts is None or ts > best_ts:
            best_ts = ts
            best_path = f

    if best_path is None:
        raise FileNotFoundError(
            f"Aucun fichier {station_cfg.name}_*.txt trouvé dans {data_dir}"
        )

    return best_path


# ============================================================
# PARTIE 3/10 : Lecture de la dernière mesure de la station
# ============================================================

def wait_until_file_complete(path: Path, timeout=2.0, check_every=0.1):
    """
    Attend que le fichier ne change plus de taille (écriture terminée).
    timeout = temps max d'attente
    """
    start = time.time()
    last_size = -1

    while time.time() - start < timeout:
        size = os.path.getsize(path)
        if size == last_size:
            return True  # stable = terminé
        last_size = size
        time.sleep(check_every)

    print(f"[WARN] Fichier {path} peut être incomplet (timeout).")
    return False

def parse_last_measurement(path: Path) -> StationMeasurement:
    """
    Lit la dernière VRAIE ligne de données du fichier 'path' en se basant sur :
      - au moins 7 colonnes
      - les 2 premières colonnes forment une date+heure valide "%d/%m/%Y %H:%M:%S"

    On parcourt le fichier de la fin vers le début pour prendre la dernière mesure valide.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    last_data_line = None
    ts = None
    power = speed_ms = dir_deg = rh = temp_c = None

    # On parcourt les lignes de la fin vers le début
    for raw in reversed(lines):
        line = raw.strip()
        if not line:
            continue

        # Découpage sur n'importe quel espace/tab
        parts = re.split(r"\s+", line)
        if len(parts) < 7:
            # trop peu de colonnes pour être une ligne de données
            continue

        # On teste si les deux premières colonnes sont une date+heure valide
        date_str = parts[0]
        time_str = parts[1]
        dt_str = f"{date_str} {time_str}"

        try:
            ts_candidate = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
        except ValueError:
            # ce n'est pas une ligne du type "dd/mm/yyyy hh:mm:ss ..."
            continue

        # Si on est là, c'est bien une ligne de données valide -> on parse les floats
        power_str = parts[2]
        speed_str = parts[3]
        dir_str = parts[4]
        rh_str = parts[5]
        temp_str = parts[6]

        def ffloat(s: str) -> float:
            return float(s.replace(",", "."))

        ts = ts_candidate
        power = ffloat(power_str)
        speed_ms = ffloat(speed_str)
        dir_deg = ffloat(dir_str)
        rh = ffloat(rh_str)
        temp_c = ffloat(temp_str)

        last_data_line = line
        break

    if last_data_line is None:
        raise ValueError(f"Aucune ligne de données valide trouvée dans {path}")

    return StationMeasurement(
        timestamp=ts,
        power=power,
        speed_ms=speed_ms,
        dir_deg=dir_deg,
        rh=rh,
        air_temp_c=temp_c,
    )

# ============================================================
# PARTIE 4/10 : Conversions vent (dir/speed) <-> (u, v)
# ============================================================

def wind_dirspeed_to_uv(speed_ms: float, dir_deg: float) -> Tuple[float, float]:
    """
    Convertit (speed, direction) → (u, v).

    Convention météo :
      - direction = direction D'ORIGINE du vent (d'où il vient),
      - 0° = vent venant du Nord, 90° = vent venant de l'Est.

    Donc :
      u (vers l'Est)  = -speed * sin(theta)
      v (vers le Nord) = -speed * cos(theta)
    """
    theta = np.deg2rad(dir_deg)
    u = -speed_ms * np.sin(theta)
    v = -speed_ms * np.cos(theta)
    return u, v

def station_speeddir_to_uv(station: StationMeasurement, station_cfg: StationConfig) -> Tuple[float, float]:
    """
    Convertit la mesure station (speed, dir) en (u,v) en respectant :
      - unités,
      - convention direction 'towards' vs 'from'.
    """
    dir_deg = station.dir_deg
    if station_cfg.wind_dir_is_towards:
        dir_deg = (dir_deg + 180.0) % 360.0
    return wind_dirspeed_to_uv(station.speed_ms, dir_deg)


def uv_to_speed_dir(u: float, v: float) -> Tuple[float, float]:
    """
    Convertit (u, v) → (speed, dir_deg) en gardant la convention météo
    (direction d'ORIGINE du vent, 0° = vent venant du Nord).
    """
    speed = np.sqrt(u * u + v * v)
    theta = np.arctan2(-u, -v)  # rad
    dir_deg = (np.rad2deg(theta) + 360.0) % 360.0
    return speed, dir_deg


# ============================================================
# PARTIE 5/10 : Appels Open-Meteo (4 coins) & bilinéaire
# ============================================================

def fetch_openmeteo_current(lat: float, lon: float) -> Dict[str, float]:
    """
    Récupère les variables "current" d'Open-Meteo à une coordonnée donnée :
      - temperature_2m (°C)
      - relative_humidity_2m (%)
      - wind_speed_10m (m/s)
      - wind_direction_10m (°)
    Remarque : on ne gère pas ici le timestamp exact ; on prend le "current".
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",  # vitesse en m/s
        "timezone": "UTC",
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    cur = data["current"]
    return {
        "temperature_2m": cur["temperature_2m"],
        "relative_humidity_2m": cur.get("relative_humidity_2m", np.nan),
        "wind_speed_10m": cur["wind_speed_10m"],
        "wind_direction_10m": cur["wind_direction_10m"],
        "time": cur["time"],  # string ISO
    }


def build_openmeteo_corners(grid: GridConfig) -> Dict[str, Dict[str, float]]:
    """
    Appelle Open-Meteo sur les 4 coins du domaine Safi :
      SW: (lat_min, lon_min)
      SE: (lat_min, lon_max)
      NW: (lat_max, lon_min)
      NE: (lat_max, lon_max)
    """
    corners = {}
    corners["SW"] = fetch_openmeteo_current(grid.lat_min, grid.lon_min)
    corners["SE"] = fetch_openmeteo_current(grid.lat_min, grid.lon_max)
    corners["NW"] = fetch_openmeteo_current(grid.lat_max, grid.lon_min)
    corners["NE"] = fetch_openmeteo_current(grid.lat_max, grid.lon_max)
    return corners


def bilinear_interpolation_field(
    grid: GridConfig,
    f_SW: float,
    f_SE: float,
    f_NW: float,
    f_NE: float,
) -> np.ndarray:
    """
    Construit un champ 2D (ny, nx) à partir de 4 valeurs aux coins du domaine,
    via interpolation bilinéaire en latitude/longitude.
    """
    ny, nx = grid.ny, grid.nx

    # Coordonnées géographiques sur la grille
    lat_vals = np.linspace(grid.lat_min, grid.lat_max, ny)
    lon_vals = np.linspace(grid.lon_min, grid.lon_max, nx)

    # Normalisation dans [0,1]
    ty = (lat_vals - grid.lat_min) / (grid.lat_max - grid.lat_min + 1e-12)
    tx = (lon_vals - grid.lon_min) / (grid.lon_max - grid.lon_min + 1e-12)

    Ty = ty[:, None]      # (ny,1)
    Tx = tx[None, :]      # (1,nx)

    # Coefficients bilinéaires
    f_00 = f_SW  # (0,0) : lat_min, lon_min
    f_10 = f_SE  # (1,0) : lat_min, lon_max
    f_01 = f_NW  # (0,1) : lat_max, lon_min
    f_11 = f_NE  # (1,1) : lat_max, lon_max

    field = (
        (1 - Tx) * (1 - Ty) * f_00
        + Tx * (1 - Ty) * f_10
        + (1 - Tx) * Ty * f_01
        + Tx * Ty * f_11
    )

    return field


def build_openmeteo_model_field(
    grid: GridConfig,
) -> ModelFields:
    """
    Construit un champ modèle (u, v, T) 2D à partir de 4 appels Open-Meteo
    sur les coins du domaine, puis interpolation bilinéaire.
    """
    corners = build_openmeteo_corners(grid)

    # Température (°C)
    T_SW = corners["SW"]["temperature_2m"]
    T_SE = corners["SE"]["temperature_2m"]
    T_NW = corners["NW"]["temperature_2m"]
    T_NE = corners["NE"]["temperature_2m"]

    # Humidité relative (%)
    RH_SW = corners["SW"].get("relative_humidity_2m", np.nan)
    RH_SE = corners["SE"].get("relative_humidity_2m", np.nan)
    RH_NW = corners["NW"].get("relative_humidity_2m", np.nan)
    RH_NE = corners["NE"].get("relative_humidity_2m", np.nan)

    # Vent (u, v) aux coins
    def u_v_corner(key: str) -> Tuple[float, float]:
        spd = corners[key]["wind_speed_10m"]
        ddeg = corners[key]["wind_direction_10m"]
        return wind_dirspeed_to_uv(spd, ddeg)

    u_SW, v_SW = u_v_corner("SW")
    u_SE, v_SE = u_v_corner("SE")
    u_NW, v_NW = u_v_corner("NW")
    u_NE, v_NE = u_v_corner("NE")

    # Interpolation bilinéaire sur la grille
    u_field = bilinear_interpolation_field(grid, u_SW, u_SE, u_NW, u_NE)
    v_field = bilinear_interpolation_field(grid, v_SW, v_SE, v_NW, v_NE)
    T_field = bilinear_interpolation_field(grid, T_SW, T_SE, T_NW, T_NE)
    RH_field = bilinear_interpolation_field(grid, RH_SW, RH_SE, RH_NW, RH_NE)

    return ModelFields(u=u_field, v=v_field, temp=T_field, rh=RH_field)

def fetch_openmeteo_hourly(lat: float, lon: float, model: str = "auto") -> Dict[str, list]:
    """
    Récupère les séries horaires (T2m, RH2m, WS10m, WD10m) pour une coordonnée.
    Retourne des listes synchronisées et la liste 'time' ISO.
    
    Args:
        lat: Latitude
        lon: Longitude
        model: Modèle météorologique à utiliser. Options:
            - "auto" (défaut, meilleur modèle disponible)
            - "ecmwf_ifs" (ECMWF IFS)
            - "gfs" (GFS/NOAA)
            - "gem" (CMC GEM)
            - "icon" (DWD ICON)
            - "metno_nordic" (MET Norway)
            - "jma_seam" (JMA)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure",
        "wind_speed_unit": "ms",
        "forecast_days": 7,     # Jusqu'à 7 jours (168h) selon le modèle
        "timezone": "UTC",
    }
    
    # Ajouter le paramètre model si spécifié (pas "auto")
    if model and model != "auto":
        params["models"] = model
    
    logger.info(f"Appel API Open-Meteo: lat={lat}, lon={lon}, model={model}")
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        data_points = len(data.get("hourly", {}).get("time", []))
        logger.info(f"✓ API Open-Meteo: {data_points} points de données reçus")
        
        return {
            "time": data["hourly"]["time"],
            "temperature_2m": data["hourly"]["temperature_2m"],
            "relative_humidity_2m": data["hourly"].get("relative_humidity_2m"),
            "wind_speed_10m": data["hourly"]["wind_speed_10m"],
            "wind_direction_10m": data["hourly"]["wind_direction_10m"],
            "surface_pressure": data["hourly"].get("surface_pressure"),
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Erreur API Open-Meteo: {e}")
        raise


def build_openmeteo_forecast_field(grid: GridConfig, hour_offset: int, model: str = "auto") -> ModelFields:
    """
    Prévision horaire spatialisée : on interroge Open-Meteo sur les 4 coins
    puis on interpole bilinéairement (u, v, T, RH) sur la grille.
    
    Args:
        grid: Configuration de la grille
        hour_offset: Décalage horaire (0 = maintenant, 1 = +1h, etc.)
        model: Modèle météorologique à utiliser (voir fetch_openmeteo_hourly)
    """
    # 1) Télécharger les séries horaires aux 4 coins
    SW = fetch_openmeteo_hourly(grid.lat_min, grid.lon_min, model)
    SE = fetch_openmeteo_hourly(grid.lat_min, grid.lon_max, model)
    NW = fetch_openmeteo_hourly(grid.lat_max, grid.lon_min, model)
    NE = fetch_openmeteo_hourly(grid.lat_max, grid.lon_max, model)

    # 2) Bornage de l'index horaire (sécurité)
    def clamp_idx(series: Dict[str, list], h: int) -> int:
        return int(max(0, min(h, len(series["time"]) - 1)))

    i_SW = clamp_idx(SW, hour_offset)
    i_SE = clamp_idx(SE, hour_offset)
    i_NW = clamp_idx(NW, hour_offset)
    i_NE = clamp_idx(NE, hour_offset)

    # 3) Extraire les valeurs aux coins pour cet horaire
    # Température
    T_SW = float(SW["temperature_2m"][i_SW])
    T_SE = float(SE["temperature_2m"][i_SE])
    T_NW = float(NW["temperature_2m"][i_NW])
    T_NE = float(NE["temperature_2m"][i_NE])

    # Humidité relative (peut être absente selon le modèle choisi par Open-Meteo)
    def get_rh(S, i):
        rh = S.get("relative_humidity_2m")
        return float(rh[i]) if rh is not None else np.nan

    RH_SW = get_rh(SW, i_SW)
    RH_SE = get_rh(SE, i_SE)
    RH_NW = get_rh(NW, i_NW)
    RH_NE = get_rh(NE, i_NE)

    # Vent → u,v aux coins
    def uv_at(S, i):
        ws = float(S["wind_speed_10m"][i])
        wd = float(S["wind_direction_10m"][i])
        return wind_dirspeed_to_uv(ws, wd)

    u_SW, v_SW = uv_at(SW, i_SW)
    u_SE, v_SE = uv_at(SE, i_SE)
    u_NW, v_NW = uv_at(NW, i_NW)
    u_NE, v_NE = uv_at(NE, i_NE)

    # 4) Interpolation bilinéaire sur la grille (comme pour le "current")
    u_field = bilinear_interpolation_field(grid, u_SW, u_SE, u_NW, u_NE)
    v_field = bilinear_interpolation_field(grid, v_SW, v_SE, v_NW, v_NE)
    T_field = bilinear_interpolation_field(grid, T_SW, T_SE, T_NW, T_NE)
    RH_field = bilinear_interpolation_field(grid, RH_SW, RH_SE, RH_NW, RH_NE)

    return ModelFields(u=u_field, v=v_field, temp=T_field, rh=RH_field)



# ============================================================
# PARTIE 6/10 : Solveur Helmholtz 2D (Dirichlet, Gauss–Seidel)
# ============================================================

def solve_helmholtz_dirichlet(
    rhs: np.ndarray,
    grid: GridConfig,
    max_iter: int = 5000,
    tol: float = 1e-5,
    verbose: bool = True,
) -> np.ndarray:
    """
    (∇² - 1/σ²) φ = rhs, Dirichlet φ=0 au bord.
    """
    ny, nx = rhs.shape
    h = grid.dx
    sigma = grid.sigma

    phi = np.zeros_like(rhs, dtype=float)
    inv_denom = 1.0 / (4.0 + (h * h) / (sigma * sigma))

    for it in range(max_iter):
        max_diff = 0.0

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                up = phi[i - 1, j]
                down = phi[i + 1, j]
                left = phi[i, j - 1]
                right = phi[i, j + 1]

                new_val = (up + down + left + right - (h * h) * rhs[i, j]) * inv_denom
                diff = abs(new_val - phi[i, j])
                if diff > max_diff:
                    max_diff = diff
                phi[i, j] = new_val

        if verbose and (it % 100 == 0 or it == max_iter - 1):
            print(f"[Helmholtz] Iter {it}, max diff = {max_diff:.3e}")

        if max_diff < tol:
            if verbose:
                print(f"[Helmholtz] Convergence atteinte à l'itération {it}, max diff = {max_diff:.3e}")
            break

    return phi


# ============================================================
# PARTIE 7/10 : RHS "Dirac discret" & fusion Helmholtz
# ============================================================

def latlon_to_ij_float(lat: float, lon: float, grid: GridConfig) -> Tuple[float, float]:
    """Indices fractionnaires (i_f, j_f) dans [0,ny-1]x[0,nx-1] pour un (lat,lon)."""
    ty = (lat - grid.lat_min) / (grid.lat_max - grid.lat_min + 1e-12)
    tx = (lon - grid.lon_min) / (grid.lon_max - grid.lon_min + 1e-12)
    i_f = np.clip(ty * (grid.ny - 1), 0.0, grid.ny - 1.0)
    j_f = np.clip(tx * (grid.nx - 1), 0.0, grid.nx - 1.0)
    return i_f, j_f

def bilinear_weights(i_f: float, j_f: float, ny: int, nx: int):
    """Coins entiers + poids bilinéaires qui somment à 1."""
    i0 = int(np.floor(i_f));  j0 = int(np.floor(j_f))
    i1 = min(i0 + 1, ny - 1); j1 = min(j0 + 1, nx - 1)
    sy = i_f - i0;            sx = j_f - j0
    w00 = (1 - sx) * (1 - sy)  # (i0,j0)
    w10 = sx       * (1 - sy)  # (i0,j1)
    w01 = (1 - sx) * sy        # (i1,j0)
    w11 = sx       * sy        # (i1,j1)
    return (i0, j0, i1, j1, w00, w10, w01, w11)

def bilinear_interp_indices(field: np.ndarray, i_f: float, j_f: float) -> float:
    """Interpolation bilinéaire dans un champ (ny,nx) aux indices fractionnaires (i_f,j_f)."""
    ny, nx = field.shape
    i0, j0, i1, j1, w00, w10, w01, w11 = bilinear_weights(i_f, j_f, ny, nx)
    return (field[i0, j0] * w00
          + field[i0, j1] * w10
          + field[i1, j0] * w01
          + field[i1, j1] * w11)

def build_rhs_for_point_source(
    delta_f: float,
    grid: GridConfig,
    station_cfg: StationConfig,
) -> np.ndarray:
    """
    Injection continue: source répartie sur les 4 cellules entourant (lat,lon) station,
    pondérée par les poids bilinéaires. L'interpolation de φ au point réel tend vers Δf.
    """
    ny, nx = grid.ny, grid.nx
    rhs = np.zeros((ny, nx), dtype=float)
    h = grid.dx
    sigma = grid.sigma

    i_f, j_f = latlon_to_ij_float(station_cfg.lat, station_cfg.lon, grid)
    i0, j0, i1, j1, w00, w10, w01, w11 = bilinear_weights(i_f, j_f, ny, nx)

    scale = -delta_f / (sigma * sigma * h * h)
    rhs[i0, j0] += w00 * scale
    rhs[i0, j1] += w10 * scale
    rhs[i1, j0] += w01 * scale
    rhs[i1, j1] += w11 * scale

    return rhs

def sample_model_at_station(model: ModelFields, station_cfg: StationConfig) -> Tuple[float, float, float]:
    """Échantillonne les champs modèle sur la cellule (i,j) de la station (utile pour debug)."""
    i, j = station_cfg.i_index, station_cfg.j_index
    return model.u[i, j], model.v[i, j], model.temp[i, j]

def fuse_station_with_model(
    station: StationMeasurement,
    model: ModelFields,
    grid: GridConfig,
    station_cfg: StationConfig,
    verbose: bool = True,
) -> FusedFields:
    """
    1) Biais continu au lat/lon exact,
    2) RHS sous-pixel,
    3) Solve Helmholtz,
    4) Renorm contre (modèle+φ) au lat/lon exact,
    5) Champs corrigés.
    """
    # Mesure station en (u,v)
    u_s, v_s = station_speeddir_to_uv(station, station_cfg)
    T_s = station.air_temp_c
    RH_s = station.rh

    # Modèle interpolé AU LAT/LON EXACT
    i_f, j_f = latlon_to_ij_float(station_cfg.lat, station_cfg.lon, grid)
    u_m = bilinear_interp_indices(model.u,   i_f, j_f)
    v_m = bilinear_interp_indices(model.v,   i_f, j_f)
    T_m = bilinear_interp_indices(model.temp, i_f, j_f)
    RH_m = bilinear_interp_indices(model.rh, i_f, j_f)

    # Biais
    delta_u = u_s - u_m
    delta_v = v_s - v_m
    delta_T = T_s - T_m
    delta_RH = RH_s - RH_m

    if verbose:
        print(f"[Fusion] Biais (u, v, T) au lat/lon station = ({delta_u:.3f}, {delta_v:.3f}, {delta_T:.3f})")

    # RHS sous-pixel
    rhs_u = build_rhs_for_point_source(delta_u, grid, station_cfg)
    rhs_v = build_rhs_for_point_source(delta_v, grid, station_cfg)
    rhs_T = build_rhs_for_point_source(delta_T, grid, station_cfg)
    rhs_RH = build_rhs_for_point_source(delta_RH, grid, station_cfg)

    # Solve Helmholtz
    phi_u = solve_helmholtz_dirichlet(rhs_u, grid, verbose=verbose)
    phi_v = solve_helmholtz_dirichlet(rhs_v, grid, verbose=verbose)
    phi_T = solve_helmholtz_dirichlet(rhs_T, grid, verbose=verbose)
    phi_RH = solve_helmholtz_dirichlet(rhs_RH, grid, verbose=verbose)

    # Renormalisation CONTINUE : (modèle + φ) interpolé doit égaler la mesure
    def renorm_to_match(meas_val: float, model_field: np.ndarray, phi_field: np.ndarray,
                        i_f: float, j_f: float) -> np.ndarray:
        m_interp   = bilinear_interp_indices(model_field, i_f, j_f)
        phi_interp = bilinear_interp_indices(phi_field,  i_f, j_f)
        target_inc = meas_val - m_interp
        if abs(phi_interp) < 1e-12:
            return phi_field  # évite blow-up si source trop faible
        s = target_inc / phi_interp
        return phi_field * s

    phi_u = renorm_to_match(u_s, model.u,   phi_u, i_f, j_f)
    phi_v = renorm_to_match(v_s, model.v,   phi_v, i_f, j_f)
    phi_T = renorm_to_match(T_s, model.temp, phi_T, i_f, j_f)
    phi_RH = renorm_to_match(RH_s, model.rh, phi_RH, i_f, j_f)

    # Champs corrigés
    u_corr = model.u + phi_u
    v_corr = model.v + phi_v
    T_corr = model.temp + phi_T
    RH_corr = model.rh + phi_RH

    return FusedFields(
        u_corr=u_corr,
        v_corr=v_corr,
        temp_corr=T_corr,
        rh_corr=RH_corr,
        phi_u=phi_u,
        phi_v=phi_v,
        phi_T=phi_T,
        phi_RH=phi_RH,
    )


# ============================================================
# PARTIE 8/10 : Conversion lat/lon -> indices de grille
# ============================================================

def latlon_to_ij(lat: float, lon: float, grid: GridConfig) -> Tuple[int, int]:
    """
    Convertit (lat, lon) en indices entiers (i, j) de la grille.
    """
    i_norm = (lat - grid.lat_min) / (grid.lat_max - grid.lat_min + 1e-12)
    j_norm = (lon - grid.lon_min) / (grid.lon_max - grid.lon_min + 1e-12)

    i_float = i_norm * (grid.ny - 1)
    j_float = j_norm * (grid.nx - 1)

    i_idx = int(np.clip(round(i_float), 0, grid.ny - 1))
    j_idx = int(np.clip(round(j_float), 0, grid.nx - 1))

    return i_idx, j_idx


# ============================================================
# PARTIE 9/10 : Visualisation (φ_u + champ de vent corrigé)
# ============================================================

def plot_phi_u(fused: FusedFields, station_cfg: StationConfig):
    """
    Affiche la correction Helmholtz φ_u avec la position de la station.
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(fused.phi_u, origin="lower", cmap="coolwarm")
    plt.colorbar(im, label="Correction φ_u")
    plt.scatter(
        [station_cfg.j_index],
        [station_cfg.i_index],
        c="black",
        marker="x",
        label="Station GP2",
    )
    plt.title("Correction Helmholtz pour u (φ_u)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_wind_field(
    fused: FusedFields,
    grid: GridConfig,
    station_cfg: StationConfig,
    station_meas: StationMeasurement,
):
    """
    Visualisation style "Windy" :
      - fond coloré = norme de la vitesse corrigée
      - flèches = vent (u_corr, v_corr) sous-échantillonné
      - croix = station GP2
    """
    u_corr = fused.u_corr
    v_corr = fused.v_corr

    speed_corr = np.sqrt(u_corr * u_corr + v_corr * v_corr)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(speed_corr, origin="lower", cmap="viridis")
    plt.colorbar(im, label="Vitesse corrigée (m/s)")

    # Sous-échantillonnage pour les flèches
    step = max(1, grid.nx // 20)  # ~20 flèches sur l'axe
    X = np.arange(0, grid.nx, step)
    Y = np.arange(0, grid.ny, step)
    XX, YY = np.meshgrid(X, Y)
    Uq = u_corr[::step, ::step]
    Vq = v_corr[::step, ::step]

    plt.quiver(XX, YY, Uq, Vq, scale=50)

    plt.scatter(
        [station_cfg.j_index],
        [station_cfg.i_index],
        c="red",
        marker="x",
        s=60,
        label="Station GP2",
    )

    plt.legend(loc="upper right")
    plt.title(f"Champ de vent fusionné – station GP2 ({station_meas.timestamp})")
    plt.tight_layout()
    plt.show()


# ============================================================
# PARTIE 10/10 : Pipeline one-shot + boucle temps réel
# ============================================================

def run_fusion_once(with_plots: bool = True):
    """
    Exécute UNE fois le pipeline complet.
    """
    # 1) Placer la station GP2 dans la grille
    i_s, j_s = latlon_to_ij(STATION_CFG.lat, STATION_CFG.lon, GRID_CFG)
    STATION_CFG.i_index = i_s
    STATION_CFG.j_index = j_s
    print(f"[RunOnce] Station GP2 placée en (i={i_s}, j={j_s}) dans la grille.")

    # 2) Trouver le dernier fichier GP2 et lire la dernière mesure
    latest_file = find_latest_gp2_file(STATION_CFG)
    print(f"[RunOnce] Dernier fichier station utilisé : {latest_file}")

    station_meas = parse_last_measurement(latest_file)
    print(f"[RunOnce] Dernière mesure GP2 à {station_meas.timestamp}:")
    print(f"         Speed = {station_meas.speed_ms:.2f} m/s, Dir = {station_meas.dir_deg:.1f}°")
    print(f"         RH = {station_meas.rh:.1f} %, Temp = {station_meas.air_temp_c:.2f} °C")

    # 3) Construire un champ modèle Open-Meteo (u, v, T) par interpolation bilinéaire
    print("[RunOnce] Appels Open-Meteo sur les 4 coins du domaine Safi...")
    model = build_openmeteo_model_field(GRID_CFG)
    print("[RunOnce] Champ modèle Open-Meteo construit.")

    # >>> Mettre le vent modèle à la hauteur de la station (loi puissance 1/7)
    alpha_h = (STATION_CFG.anemometer_height_m / max(STATION_CFG.model_wind_height_m, 1e-6)) ** (1.0 / 7.0)
    model.u *= alpha_h
    model.v *= alpha_h

    # 4) Fusion station + modèle via Helmholtz
    fused = fuse_station_with_model(
        station=station_meas,
        model=model,
        grid=GRID_CFG,
        station_cfg=STATION_CFG,
        verbose=True,
    )

    # 5) (optionnel) Visualisation
    if with_plots:
        plot_phi_u(fused, STATION_CFG)
        plot_wind_field(fused, GRID_CFG, STATION_CFG, station_meas)

    # 6) Vérification CONTINUE au lat/lon exact de la station
    i_f, j_f = latlon_to_ij_float(STATION_CFG.lat, STATION_CFG.lon, GRID_CFG)
    u_corr_at_station = bilinear_interp_indices(fused.u_corr, i_f, j_f)
    v_corr_at_station = bilinear_interp_indices(fused.v_corr, i_f, j_f)
    T_corr_at_station = bilinear_interp_indices(fused.temp_corr, i_f, j_f)
    RH_corr_at_station = bilinear_interp_indices(fused.rh_corr, i_f, j_f)

    # Utiliser la même convention que dans la fusion
    u_s, v_s = station_speeddir_to_uv(station_meas, STATION_CFG)
    speed_corr, dir_corr = uv_to_speed_dir(u_corr_at_station, v_corr_at_station)

    print("\n[RunOnce] Vérification (interpolation bilinéaire au lat/lon station) :")
    print(f"  u_station       = {u_s:.3f}, u_corrigé = {u_corr_at_station:.3f}")
    print(f"  v_station       = {v_s:.3f}, v_corrigé = {v_corr_at_station:.3f}")
    print(f"  |V|_station     = {station_meas.speed_ms:.3f} m/s, |V|_corrigée = {speed_corr:.3f} m/s")
    print(f"  Dir_station     = {station_meas.dir_deg:.1f}°, Dir_corrigée = {dir_corr:.1f}°")
    print(f"  T_station (°C)  = {station_meas.air_temp_c:.3f}, T_corrigée = {T_corr_at_station:.3f}")
    print(f"  RH_station (%)  = {station_meas.rh:.3f}, RH_corrigée = {RH_corr_at_station:.3f}")

    return station_meas, model, fused


def run_realtime(with_plots: bool = False, check_interval: float = 0.5):
    """
    Boucle temps réel réactive.
    """
    print("[Realtime] Surveillance du dossier data/ (réactivité maximale).")

    last_seen_file = None

    while True:
        try:
            latest = find_latest_gp2_file(STATION_CFG)

            # Premier passage
            if last_seen_file is None:
                last_seen_file = latest
                print(f"[Realtime] Fichier initial : {latest}")
                run_fusion_once(with_plots=with_plots)

            # Nouveau fichier détecté
            elif latest != last_seen_file:
                print(f"[Realtime] Nouveau fichier détecté : {latest}")

                # ⏳ On attend que l’écriture soit terminée
                wait_until_file_complete(latest)

                last_seen_file = latest
                run_fusion_once(with_plots=with_plots)

        except Exception as e:
            print(f"[Realtime] Erreur : {e}")

        # Petit sleep pour éviter 100% CPU
        time.sleep(check_interval)



if __name__ == "__main__":
    # 🔹 Mode "one-shot" (une seule exécution avec affichage complet)
    # run_fusion_once(with_plots=True)

    # 🔹 Mode temps réel (rafraîchissement toutes les 60 s, sans plots par défaut)
    run_realtime(with_plots=False)
