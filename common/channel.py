"""Shared channel model (co-owned by Student A and Student B).

This module is the canonical import surface for the physical-layer channel
model. To avoid forking the physics into two places, it re-exports the core
functions from the original reference simulator
(``project/uav_extended_simulation.py``).

Both students should import from here::

    from common.channel import compute_user_rate, has_line_of_sight

Any agreed improvement to ``compute_user_rate`` (e.g. the Section 7 fix to use
the full channel model) lands in the reference simulator and is picked up here
automatically.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# --- Locate and import the reference simulator as a module ------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_PATH = _REPO_ROOT / "project" / "uav_extended_simulation.py"

if not _SIM_PATH.exists():  # pragma: no cover - defensive
    raise FileNotFoundError(
        f"Reference simulator not found at {_SIM_PATH}. "
        "common.channel re-exports the physics from it."
    )

_spec = importlib.util.spec_from_file_location("uav_extended_simulation", _SIM_PATH)
_sim = importlib.util.module_from_spec(_spec)
sys.modules["uav_extended_simulation"] = _sim
_spec.loader.exec_module(_sim)  # type: ignore[union-attr]

# --- Re-export the physics API ---------------------------------------------
# Radio constants
speed_of_light = _sim.speed_of_light
carrier_frequency_hz = _sim.carrier_frequency_hz
signal_wavelength = _sim.signal_wavelength
channel_bandwidth_hz = _sim.channel_bandwidth_hz
noise_power_watts = _sim.noise_power_watts
AG_ENV = _sim.AG_ENV
DEFAULT_ENV = _sim.DEFAULT_ENV

# Unit conversion
dbm_to_watts = _sim.dbm_to_watts
watts_to_dbm = _sim.watts_to_dbm

# Geometry
distance_3d = _sim.distance_3d
elevation_angle_deg = _sim.elevation_angle_deg

# Antenna
dipole_gain_dBi = _sim.dipole_gain_dBi
antenna_gain_dBi = _sim.antenna_gain_dBi

# Channel
fspl_dB = _sim.fspl_dB
los_probability = _sim.los_probability
air_to_ground_path_loss_dB = _sim.air_to_ground_path_loss_dB
compute_snr_full = _sim.compute_snr_full
compute_capacity_full = _sim.compute_capacity_full

# Geometry-aware LoS + the rate function the surrogate will learn to replace
has_line_of_sight = _sim.has_line_of_sight
compute_user_rate = _sim.compute_user_rate

# Optimizer (Section 5) — surrogate gets plugged in here in Step 11
optimize_uav_3d = _sim.optimize_uav_3d

__all__ = [
    "speed_of_light",
    "carrier_frequency_hz",
    "signal_wavelength",
    "channel_bandwidth_hz",
    "noise_power_watts",
    "AG_ENV",
    "DEFAULT_ENV",
    "dbm_to_watts",
    "watts_to_dbm",
    "distance_3d",
    "elevation_angle_deg",
    "dipole_gain_dBi",
    "antenna_gain_dBi",
    "fspl_dB",
    "los_probability",
    "air_to_ground_path_loss_dB",
    "compute_snr_full",
    "compute_capacity_full",
    "has_line_of_sight",
    "compute_user_rate",
    "optimize_uav_3d",
]
