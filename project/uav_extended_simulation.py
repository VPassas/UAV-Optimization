import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.special import erfc
from scipy.optimize import brentq, minimize, differential_evolution

warnings.filterwarnings('ignore')

try:
    import osmnx as ox
    import pandas as pd
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

# ============================================================
# UAV Extended Communication Simulation  (v2 — Enhanced)
# Sections:
#   0. Shared utilities, full channel model, antenna functions
#   1. BER vs SNR — multiple modulations
#   2. Modulation comparison + energy efficiency
#   3. 3D coordinate geometry
#   4. Antenna pattern — narrow cosine vs half-wave dipole
#   5. Real OSM town layout + constrained 3D UAV optimization
#   6. Fair resource allocation (full channel model)
#   7. UAV-to-vehicle communication & trajectory
#   8. Energy efficiency analysis
#   9. Optimization technique comparison
# ============================================================


# =============================================================
# SECTION 0: Shared Utilities
# =============================================================

# --- Global radio parameters ---
speed_of_light      = 3e8          # m/s
carrier_frequency_hz = 2e9         # 2 GHz carrier
signal_wavelength   = speed_of_light / carrier_frequency_hz   # ~0.15 m
channel_bandwidth_hz = 1e6         # 1 MHz channel
kT_dBm_per_hz       = -174         # thermal noise floor (dBm/Hz)
noise_figure_dB     = 7            # receiver noise figure (dB)

# Legacy log-distance environment presets (used in Sections 1–3)
ENV_PRESETS = {
    "Rural":      {"n": 2.2, "sigma_dB": 3.0,  "fading": "Rician",   "K_dB": 8},
    "Urban":      {"n": 2.7, "sigma_dB": 6.0,  "fading": "Rician",   "K_dB": 5},
    "DenseUrban": {"n": 3.2, "sigma_dB": 8.0,  "fading": "Rayleigh", "K_dB": None},
    "VeryDense":  {"n": 3.6, "sigma_dB": 10.0, "fading": "Rayleigh", "K_dB": None},
    "Forest":     {"n": 3.0, "sigma_dB": 8.0,  "fading": "Rayleigh", "K_dB": None},
}

# Al-Hourani (2014) air-to-ground channel model parameters.
# Reference: A. Al-Hourani, S. Kandeepan, S. Lardner, "Optimal LAP Altitude for
# Maximum Coverage," IEEE Wireless Commun. Letters, vol. 3, no. 6, pp. 569–572, 2014.
# a, b  : shape parameters of the LoS probability sigmoid curve.
# eta_los / eta_nlos : mean excess path loss (dB) for LoS and NLoS conditions.
# sigma_dB : standard deviation of log-normal shadowing.
AG_ENV = {
    "Suburban":   {"a":  4.88, "b": 0.43, "eta_los":  0.1, "eta_nlos": 21.0, "sigma_dB": 3.0},
    "Urban":      {"a":  9.61, "b": 0.16, "eta_los":  1.0, "eta_nlos": 20.0, "sigma_dB": 6.0},
    "DenseUrban": {"a": 12.08, "b": 0.11, "eta_los":  1.6, "eta_nlos": 23.0, "sigma_dB": 8.0},
    "Highrise":   {"a": 27.23, "b": 0.08, "eta_los":  2.3, "eta_nlos": 34.0, "sigma_dB": 10.0},
}
DEFAULT_ENV = "Urban"


# --- Unit conversion ---
def dbm_to_watts(dbm):
    return 10 ** ((np.asarray(dbm, dtype=float) - 30) / 10)

def watts_to_dbm(w):
    return 10 * np.log10(np.maximum(np.asarray(w, dtype=float), 1e-30)) + 30


# --- Noise ---
def compute_noise_power_watts(bandwidth_hz=1e6, nf_dB=7):
    # Total noise power = kT (dBm/Hz) + 10·log10(BW) + noise figure
    noise_dBm = kT_dBm_per_hz + 10 * np.log10(bandwidth_hz) + nf_dB
    return dbm_to_watts(noise_dBm)

noise_power_watts = compute_noise_power_watts(channel_bandwidth_hz, noise_figure_dB)


# --- Path loss ---
def fspl_dB(distance_m, wavelength_m=None):
    # Free-Space Path Loss: increases 6 dB per doubling of distance.
    if wavelength_m is None:
        wavelength_m = signal_wavelength
    return 20 * np.log10(4 * np.pi * np.maximum(distance_m, 1e-6) / wavelength_m)

def log_distance_path_loss_dB(distance_m, wavelength_m, n, sigma_dB, rng):
    # Log-distance model: FSPL at 1 m reference + exponent term + shadowing.
    d0 = 1.0
    pl0 = fspl_dB(d0, wavelength_m)
    shadowing = rng.normal(0.0, sigma_dB, size=np.shape(distance_m))
    return pl0 + 10 * n * np.log10(np.maximum(distance_m, 1e-6) / d0) + shadowing


# --- Small-scale fading ---
def fading_power_gain(num_samples, fading_type="Rayleigh", K_dB=None, rng=None):
    # Rayleigh: no dominant path (fully scattered signal, exponential power).
    # Rician:   dominant LoS path exists; K_dB is the ratio of LoS to scattered power.
    rng = np.random.default_rng() if rng is None else rng
    if fading_type == "Rayleigh":
        return rng.exponential(scale=1.0, size=num_samples)
    if fading_type == "Rician":
        K = 10 ** (K_dB / 10)
        s     = np.sqrt(K / (K + 1))
        sigma = np.sqrt(1 / (2 * (K + 1)))
        x = rng.normal(loc=s,   scale=sigma, size=num_samples)
        y = rng.normal(loc=0.0, scale=sigma, size=num_samples)
        return x**2 + y**2
    raise ValueError(f"Unknown fading type: {fading_type}")


# --- Geometry ---
def distance_3d(pos_a, pos_b):
    a = np.asarray(pos_a, dtype=float)
    b = np.asarray(pos_b, dtype=float)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))

def elevation_angle_deg(uav_pos, ground_pos):
    """Elevation angle from ground user to UAV (0° = horizontal, 90° = directly overhead)."""
    d_horiz = np.sqrt((uav_pos[0] - ground_pos[0])**2 + (uav_pos[1] - ground_pos[1])**2)
    dz = abs(uav_pos[2] - ground_pos[2])
    return np.degrees(np.arctan2(dz, np.maximum(d_horiz, 1e-9)))


# --- Antenna models ---
def antenna_gain_dBi(elevation_deg, hpbw_deg=60.0):
    """
    Narrow-beam cosine-power antenna (HPBW = 60°, boresight = nadir).
    Kept as a baseline reference for comparison against the dipole.
    """
    elevation_deg = np.asarray(elevation_deg, dtype=float)
    theta_off = 90.0 - elevation_deg          # angle from boresight (0 = straight down)
    half_angle = hpbw_deg / 2.0
    n_ant = -3.0 / (10 * np.log10(np.cos(np.radians(half_angle))**2 + 1e-30))
    theta_rad = np.radians(np.clip(theta_off, 0, 89.9))
    gain_lin = np.cos(theta_rad) ** n_ant
    return 10 * np.log10(np.maximum(gain_lin, 1e-10))

def dipole_gain_dBi(elevation_deg):
    """
    Half-wave dipole gain pattern (wide beam, HPBW ≈ 78°, omnidirectional in azimuth).
    For a horizontal dipole mounted on the UAV, the elevation-plane pattern is:
        G(φ) = 1.641 · [cos(π/2 · sin(φ)) / cos(φ)]²
    where φ is the elevation angle from the horizontal plane.

    This is the recommended antenna for UAV communications because:
    - Wide beam covers a large ground footprint without mechanical steering.
    - Near-omnidirectional in azimuth (no need to track individual users).
    - Lightweight, simple, well-characterised.

    Reference: C. A. Balanis, "Antenna Theory: Analysis and Design," 4th ed.,
    Wiley, 2016, Ch. 4 — half-wave dipole radiation pattern.
    """
    elev_rad   = np.radians(np.asarray(elevation_deg, dtype=float))
    cos_e      = np.cos(elev_rad)
    cos_e_safe = np.where(np.abs(cos_e) < 1e-6, 1e-6, cos_e)
    numerator  = np.cos(np.pi / 2 * np.sin(elev_rad))
    gain_lin   = 1.641 * (numerator / cos_e_safe) ** 2
    return 10 * np.log10(np.maximum(gain_lin, 1e-10))


# --- Al-Hourani air-to-ground channel model ---
def los_probability(elevation_deg, env_key=DEFAULT_ENV):
    """
    Probability of LoS as a sigmoid function of elevation angle.
    Higher elevation → more likely to have a clear line of sight.
    Reference: Al-Hourani et al., IEEE Wireless Commun. Letters, 2014.
    """
    env = AG_ENV[env_key]
    a, b = env["a"], env["b"]
    return 1.0 / (1.0 + a * np.exp(-b * (elevation_deg - a)))

def air_to_ground_path_loss_dB(uav_pos, user_pos, env_key=DEFAULT_ENV, rng=None):
    """
    Full UAV air-to-ground path loss (Al-Hourani 2014) including:
    - FSPL as the distance-dependent reference
    - P_LoS-weighted excess attenuation (eta_LoS vs eta_NLoS)
    - Log-normal shadowing (large-scale fading)

    Returns (path_loss_dB, p_los).
    Small-scale fading is applied separately in compute_snr_full().
    """
    if rng is None:
        rng = np.random.default_rng()
    env  = AG_ENV[env_key]
    dist = distance_3d(uav_pos, user_pos)
    elev = elevation_angle_deg(uav_pos, user_pos)
    p_los   = los_probability(elev, env_key)
    pl_fs   = fspl_dB(dist)
    # Expected excess loss = P_LoS·η_LoS + (1−P_LoS)·η_NLoS
    pl_mean = pl_fs + p_los * env["eta_los"] + (1 - p_los) * env["eta_nlos"]
    shadow  = rng.normal(0.0, env["sigma_dB"])   # log-normal shadowing
    return pl_mean + shadow, p_los

def compute_snr_full(uav_pos, user_pos, p_tx_watts, env_key=DEFAULT_ENV,
                     rng=None, apply_fading=True):
    """
    Instantaneous SNR with the complete channel stack:
    Al-Hourani path loss + log-normal shadowing + Rician/Rayleigh fading + dipole gain.
    Rician fading is used when p_los > 0.5; Rayleigh otherwise.
    """
    if rng is None:
        rng = np.random.default_rng()
    elev       = elevation_angle_deg(uav_pos, user_pos)
    g_ant_lin  = 10 ** (dipole_gain_dBi(elev) / 10)
    pl_dB, p_los = air_to_ground_path_loss_dB(uav_pos, user_pos, env_key, rng)
    pl_lin     = 10 ** (pl_dB / 10)
    p_rx       = p_tx_watts * g_ant_lin / pl_lin
    if apply_fading:
        if p_los > 0.5:
            # K factor scales with LoS dominance (0 – 15 dB)
            K_dB   = float(np.clip(10 * np.log10(p_los / max(1 - p_los, 1e-9)), 0, 15))
            fading = fading_power_gain(1, "Rician", K_dB=K_dB, rng=rng)[0]
        else:
            fading = fading_power_gain(1, "Rayleigh", rng=rng)[0]
        p_rx *= fading
    return p_rx / noise_power_watts

def compute_capacity_full(uav_pos, user_pos, p_tx_watts, env_key=DEFAULT_ENV,
                           rng=None, n_mc=30):
    """
    Ergodic (average) capacity via Monte Carlo over small-scale fading realizations.
    Averaging over n_mc independent fading samples gives a stable estimate.
    Returns capacity in bits/s/Hz.
    """
    if rng is None:
        rng = np.random.default_rng()
    caps = [np.log2(1 + max(compute_snr_full(uav_pos, user_pos, p_tx_watts,
                                              env_key, rng, apply_fading=True), 0))
            for _ in range(n_mc)]
    return float(np.mean(caps))


# --- Energy efficiency ---
def energy_efficiency(capacity_bps_hz, p_tx_watts, p_circuit_watts=0.1):
    """
    Energy Efficiency (EE) = capacity / total power  [bits / Joule / Hz].
    p_circuit_watts: static power draw of UAV electronics (default 100 mW).
    Reference: G. Auer et al., "How much energy is needed to run a wireless network?"
    IEEE Wireless Commun., vol. 18, no. 5, pp. 40–49, 2011.
    """
    return capacity_bps_hz / (p_tx_watts + p_circuit_watts)


# =============================================================
# SECTION 1: BER vs SNR — Multiple Modulation Techniques
# =============================================================

# M = constellation size. More points per symbol = faster data, higher SNR required.
MODULATIONS = {
    "BPSK":   2,    # 1 bit/symbol  — most robust
    "QPSK":   4,    # 2 bits/symbol
    "16-QAM": 16,   # 4 bits/symbol
    "64-QAM": 64,   # 6 bits/symbol — highest speed, needs strong signal
}

def theoretical_ber(snr_linear, M):
    """Standard closed-form BER approximation for M-ary QAM over AWGN (Gray coded)."""
    snr_linear = np.asarray(snr_linear, dtype=float)
    if M == 2:
        return 0.5 * erfc(np.sqrt(snr_linear))
    k = np.log2(M)
    return (2 * (1 - 1 / np.sqrt(M)) / k) * erfc(
        np.sqrt(3 * k * snr_linear / (2 * (M - 1)))
    )

def plot_ber_vs_snr():
    """BER vs SNR for BPSK/QPSK/16-QAM/64-QAM at three UAV–user distances."""
    snr_dB  = np.linspace(-5, 35, 500)
    snr_lin = 10 ** (snr_dB / 10)
    ground_user   = (0, 0, 0)
    uav_positions = [
        ("UAV at (0,0,100) — d=100m",   (0,   0, 100)),
        ("UAV at (200,0,100) — d=224m", (200,  0, 100)),
        ("UAV at (400,0,150) — d=427m", (400,  0, 150)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors    = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for ax, (label, uav_pos) in zip(axes, uav_positions):
        dist = distance_3d(uav_pos, ground_user)
        pl   = fspl_dB(dist)
        for (mod_name, M), col in zip(MODULATIONS.items(), colors):
            ber = np.clip(theoretical_ber(snr_lin, M), 1e-8, 1)
            ax.semilogy(snr_dB, ber, label=mod_name, color=col, lw=2)
        ax.axhline(1e-3, color='gray', ls='--', lw=0.8, label="BER=10⁻³")
        ax.axhline(1e-5, color='gray', ls=':',  lw=0.8, label="BER=10⁻⁵")
        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"{label}\nFSPL = {pl:.1f} dB")
        ax.set_ylim(1e-7, 1)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Bit Error Rate (BER)")
    fig.suptitle("BER vs SNR — Multiple Modulations at Different UAV Positions", fontsize=13)
    plt.tight_layout()


# =============================================================
# SECTION 2: Modulation Comparison + Energy Efficiency
# =============================================================

def required_snr_dB_for_ber(M, target_ber):
    """Minimum SNR (dB) to achieve target_ber for M-QAM, found via Brent's method."""
    def f(snr_dB):
        return theoretical_ber(10 ** (snr_dB / 10), M) - target_ber
    return brentq(f, -5, 50)

def plot_modulation_comparison(target_ber=1e-4):
    """
    Three-panel comparison at a fixed BER target:
    1. Required SNR per modulation.
    2. Required transmit power (dBm).
    3. Energy efficiency at the minimum required power.
    """
    mod_names        = list(MODULATIONS.keys())
    Ms               = list(MODULATIONS.values())
    snr_required_dB  = [required_snr_dB_for_ber(M, target_ber) for M in Ms]
    ref_dist         = np.sqrt(200**2 + 100**2)   # UAV at 100 m alt, user 200 m horizontal
    pl               = fspl_dB(ref_dist)
    ptx_required_dBm = [snr + 10 * np.log10(noise_power_watts) + 30 + pl
                        for snr in snr_required_dB]
    spectral_eff     = [np.log2(M) for M in Ms]
    ee_vals          = [energy_efficiency(se, dbm_to_watts(ptx))
                        for se, ptx in zip(spectral_eff, ptx_required_dBm)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x      = np.arange(len(mod_names))
    colors = ["steelblue", "coral", "mediumseagreen"]

    axes[0].bar(x, snr_required_dB, color=colors[0])
    for i, (se, snr) in enumerate(zip(spectral_eff, snr_required_dB)):
        axes[0].text(i, snr + 0.3, f"{se:.0f} b/s/Hz", ha='center', fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(mod_names)
    axes[0].set_ylabel("Required SNR (dB)")
    axes[0].set_title(f"Required SNR @ BER = {target_ber:.0e}")
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x, ptx_required_dBm, color=colors[1])
    for i, ptx in enumerate(ptx_required_dBm):
        axes[1].text(i, ptx + 0.2, f"{ptx:.1f} dBm", ha='center', fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(mod_names)
    axes[1].set_ylabel("Required Tx Power (dBm)")
    axes[1].set_title("Required Transmit Power")
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].bar(x, ee_vals, color=colors[2])
    for i, ee in enumerate(ee_vals):
        axes[2].text(i, ee + max(ee_vals) * 0.01, f"{ee:.2f}", ha='center', fontsize=8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(mod_names)
    axes[2].set_ylabel("Energy Efficiency (bits/J/Hz)")
    axes[2].set_title("Energy Efficiency at Minimum Tx Power")
    axes[2].grid(axis='y', alpha=0.3)

    fig.suptitle(f"Modulation Comparison @ BER = {target_ber:.0e},  d = {ref_dist:.0f} m",
                 fontsize=13)
    plt.tight_layout()


# =============================================================
# SECTION 3: 3D Coordinate Geometry
# =============================================================

def plot_3d_geometry():
    """3D and top-down view of the UAV-to-ground link geometry."""
    uav   = np.array([0, 0, 100])
    users = {
        "Legit User":   np.array([ 200,  50, 0]),
        "Eavesdropper": np.array([ 500, 100, 0]),
        "Car 1":        np.array([ 150, -80, 0]),
        "Car 2":        np.array([-100, 200, 0]),
    }
    fig  = plt.figure(figsize=(14, 5))
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.scatter(*uav, s=200, marker='^', color='red', zorder=5, label='UAV')
    for name, pos in users.items():
        ax3d.scatter(*pos, s=80, marker='o', zorder=5)
        ax3d.plot([uav[0], pos[0]], [uav[1], pos[1]], [uav[2], pos[2]], 'k--', alpha=0.5)
        dist = distance_3d(uav, pos)
        elev = elevation_angle_deg(uav, pos)
        ax3d.text(pos[0], pos[1], pos[2] + 8,
                  f"{name}\nd={dist:.0f}m  θ={elev:.1f}°", fontsize=7)
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Link Geometry")

    ax2d = fig.add_subplot(122)
    ax2d.plot(uav[0], uav[1], 'r^', ms=14, label=f"UAV z={uav[2]}m")
    for name, pos in users.items():
        ax2d.plot(pos[0], pos[1], 'o', ms=8)
        ax2d.plot([uav[0], pos[0]], [uav[1], pos[1]], 'k--', alpha=0.4)
        h_dist = np.sqrt((uav[0]-pos[0])**2 + (uav[1]-pos[1])**2)
        ax2d.annotate(f"{name}\nh={h_dist:.0f}m", (pos[0], pos[1]),
                      textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax2d.set_xlabel("X (m)"); ax2d.set_ylabel("Y (m)")
    ax2d.set_title("Top-Down Projection")
    ax2d.legend(fontsize=8); ax2d.grid(True, alpha=0.3); ax2d.set_aspect('equal')
    plt.tight_layout()


# =============================================================
# SECTION 4: Antenna Pattern — Narrow Cosine vs Half-Wave Dipole
# =============================================================

def plot_antenna_pattern():
    """
    Compare the narrow 60° cosine beam (old) against the half-wave dipole (new).
    Three panels: polar pattern overlay, narrow-beam ground footprint, dipole footprint.
    The dipole's wider beam covers a larger area at the cost of slightly lower peak gain.
    """
    fig = plt.figure(figsize=(18, 5))

    # --- Polar comparison ---
    ax_p = fig.add_subplot(131, projection='polar')
    theta_deg  = np.linspace(0, 90, 500)   # off-boresight angle from nadir
    elevation  = 90 - theta_deg
    ax_p.plot(np.radians(theta_deg), antenna_gain_dBi(elevation),
              'b-', lw=2, label='Narrow cosine (60° HPBW)')
    ax_p.plot(np.radians(theta_deg), dipole_gain_dBi(elevation),
              'r-', lw=2, label='Half-wave dipole (~78° HPBW)')
    ax_p.set_theta_zero_location("N"); ax_p.set_theta_direction(-1)
    ax_p.set_thetamin(0); ax_p.set_thetamax(90)
    ax_p.set_title("Elevation Gain Pattern (dBi)", pad=20)
    ax_p.legend(loc='lower right', fontsize=8)

    # --- Ground footprints ---
    uav_alt   = 100
    grid_size = 500
    xs = np.linspace(-grid_size, grid_size, 200)
    ys = np.linspace(-grid_size, grid_size, 200)
    X, Y      = np.meshgrid(xs, ys)
    h_dist    = np.sqrt(X**2 + Y**2)
    dist_3d_g = np.sqrt(h_dist**2 + uav_alt**2)
    elev_map  = np.degrees(np.arctan2(uav_alt, np.maximum(h_dist, 1e-6)))
    pl_map    = fspl_dB(dist_3d_g)
    p_tx_dBm  = 20

    for idx, (gain_fn, title) in enumerate([
        (antenna_gain_dBi, "Narrow Cosine (60° HPBW)"),
        (dipole_gain_dBi,  "Half-Wave Dipole (~78° HPBW)"),
    ]):
        ax = fig.add_subplot(132 + idx)
        prx = p_tx_dBm - pl_map + gain_fn(elev_map)
        im  = ax.pcolormesh(X, Y, prx, cmap='jet', shading='auto', vmin=-80, vmax=-30)
        fig.colorbar(im, ax=ax, label="Rx Power (dBm)")
        ax.set_title(f"{title}\nUAV alt={uav_alt}m, Ptx={p_tx_dBm}dBm")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect('equal')

    fig.suptitle("Antenna Comparison: Narrow Directional vs Wide-Beam Half-Wave Dipole",
                 fontsize=13)
    plt.tight_layout()


# =============================================================
# SECTION 5: Real OSM Town Layout + Constrained 3D Optimization
# =============================================================

def generate_town(area_size=500, num_buildings=15, num_users=20, seed=42):
    """Synthetic fallback: random box buildings + users placed outside buildings."""
    rng = np.random.default_rng(seed)
    buildings = []
    for _ in range(num_buildings):
        cx = rng.uniform(50, area_size - 50)
        cy = rng.uniform(50, area_size - 50)
        w  = rng.uniform(15, 50)
        d  = rng.uniform(15, 50)
        h  = rng.uniform(10, 50)
        buildings.append((cx, cy, w, d, h))

    def inside_building(px, py):
        return any(abs(px - cx) < w / 2 and abs(py - cy) < d / 2
                   for cx, cy, w, d, _ in buildings)

    users = []
    while len(users) < num_users:
        px = rng.uniform(0, area_size)
        py = rng.uniform(0, area_size)
        if not inside_building(px, py):
            users.append((px, py, 0.0))
    return buildings, np.array(users), []   # empty roads list for API consistency


def generate_town_osm(place="Ermoupoli, Syros, Greece", area_size=500,
                       num_users=20, seed=42):
    """
    Fetch real building footprints and road network from OpenStreetMap (osmnx).
    Reprojects to a local UTM metre coordinate system centred on the town.
    Users are placed on road segments rather than randomly, matching reality.

    Falls back to synthetic generation if osmnx is unavailable or the fetch fails.
    Install with:  pip install osmnx

    Reference: G. Boeing, "OSMnx: New methods for acquiring, constructing, analyzing,
    and visualizing complex street networks," Computers, Environment and Urban Systems,
    vol. 65, pp. 126–139, 2017.
    """
    if not OSMNX_AVAILABLE:
        print("[INFO] osmnx not installed — using synthetic town.  "
              "Install: pip install osmnx")
        return generate_town(area_size=area_size, num_users=num_users, seed=seed)

    try:
        ox.settings.log_console = False
        ox.settings.use_cache   = True
        print(f"[OSM] Fetching '{place}' ...")

        # --- Buildings ---
        gdf_b = ox.features_from_place(place, tags={'building': True})
        gdf_b = gdf_b[gdf_b.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        gdf_b = gdf_b.to_crs(gdf_b.estimate_utm_crs())

        centroid = gdf_b.geometry.unary_union.centroid
        cx0 = centroid.x - area_size / 2
        cy0 = centroid.y - area_size / 2

        buildings = []
        for _, row in gdf_b.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            bx = (minx + maxx) / 2 - cx0
            by = (miny + maxy) / 2 - cy0
            bw = maxx - minx
            bd = maxy - miny
            if not (0 < bx < area_size and 0 < by < area_size and bw > 1 and bd > 1):
                continue
            # Height from OSM tag, or estimate from building:levels × 3 m
            bh = 10.0
            for attr in ['height', 'building:levels']:
                val = row.get(attr)
                if val is not None and pd.notna(val):
                    try:
                        bh = float(str(val).replace('m', '').strip())
                        if attr == 'building:levels':
                            bh *= 3.0
                        break
                    except (ValueError, TypeError):
                        pass
            buildings.append((bx, by, bw, bd, bh))

        # --- Roads ---
        G_road = ox.graph_from_place(place, network_type='drive')
        G_road = ox.projection.project_graph(G_road, to_crs=gdf_b.crs)
        _, edges = ox.graph_to_gdfs(G_road)
        roads = []
        for _, edge in edges.iterrows():
            pts = [(x - cx0, y - cy0) for x, y in edge.geometry.coords]
            if any(0 <= px <= area_size and 0 <= py <= area_size for px, py in pts):
                roads.append(pts)

        # --- Users placed on roads ---
        rng = np.random.default_rng(seed)
        road_pts = [(px, py) for road in roads for px, py in road
                    if 0 < px < area_size and 0 < py < area_size]
        users = []
        attempts = 0
        while len(users) < num_users and attempts < 20000:
            attempts += 1
            if road_pts:
                px, py = road_pts[rng.integers(len(road_pts))]
                px += rng.normal(0, 5)
                py += rng.normal(0, 5)
            else:
                px = rng.uniform(0, area_size)
                py = rng.uniform(0, area_size)
            px = float(np.clip(px, 0, area_size))
            py = float(np.clip(py, 0, area_size))
            if not any(abs(px - bx) < bw / 2 and abs(py - by) < bd / 2
                       for bx, by, bw, bd, _ in buildings):
                users.append((px, py, 0.0))

        print(f"[OSM] {len(buildings)} buildings, {len(roads)} road segments, "
              f"{len(users)} users.")
        return buildings, np.array(users), roads

    except Exception as exc:
        print(f"[WARN] OSM fetch failed ({exc}) — using synthetic town.")
        return generate_town(area_size=area_size, num_users=num_users, seed=seed)


def has_line_of_sight(uav_pos, user_pos, buildings):
    """
    Ray-AABB slab intersection test.
    Casts the ray from uav_pos to user_pos and checks every building's bounding box.
    Returns True when no building blocks the path.
    """
    p0 = np.array(uav_pos, dtype=float)
    p1 = np.array(user_pos, dtype=float)
    d  = p1 - p0
    for cx, cy, w, d_b, h in buildings:
        x_min, x_max = cx - w / 2,   cx + w / 2
        y_min, y_max = cy - d_b / 2, cy + d_b / 2
        z_min, z_max = 0.0, h
        t_min, t_max = 0.0, 1.0
        for axis, (lo, hi) in enumerate([(x_min, x_max), (y_min, y_max), (z_min, z_max)]):
            if abs(d[axis]) < 1e-12:
                if p0[axis] < lo or p0[axis] > hi:
                    break
            else:
                t1 = (lo - p0[axis]) / d[axis]
                t2 = (hi - p0[axis]) / d[axis]
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    break
        else:
            if t_min <= t_max:
                return False
    return True


def compute_user_rate(uav_pos, user_pos, p_tx_watts, env_key=DEFAULT_ENV,
                      buildings=None, rng=None, n_mc=20):
    """
    Ergodic capacity (bits/s/Hz) using the full Al-Hourani channel model.

    When a building list is provided the function performs a geometric LoS check
    and applies the appropriate excess attenuation (eta_los or eta_nlos) directly,
    then averages over n_mc fading samples (Monte Carlo).

    Without buildings, it uses the purely probabilistic LoS model.
    """
    if rng is None:
        rng = np.random.default_rng()

    if buildings is not None:
        los_geo = has_line_of_sight(uav_pos, user_pos, buildings)
        env     = AG_ENV[env_key]
        dist    = distance_3d(uav_pos, user_pos)
        elev    = elevation_angle_deg(uav_pos, user_pos)
        g_lin   = 10 ** (dipole_gain_dBi(elev) / 10)
        pl_fs   = fspl_dB(dist)
        if los_geo:
            pl_excess = env["eta_los"]
            ftype, K_dB = "Rician", 8.0
        else:
            pl_excess = env["eta_nlos"]
            ftype, K_dB = "Rayleigh", None
        shadow      = rng.normal(0, env["sigma_dB"])
        pl_dB       = pl_fs + pl_excess + shadow
        p_rx_mean   = p_tx_watts * g_lin / (10 ** (pl_dB / 10))
        caps = [np.log2(1 + max(p_rx_mean * fading_power_gain(1, ftype, K_dB, rng)[0]
                                / noise_power_watts, 0))
                for _ in range(n_mc)]
        return float(np.mean(caps))
    else:
        return compute_capacity_full(uav_pos, user_pos, p_tx_watts, env_key, rng, n_mc)


# --- Optimization objective wrapper ---
def _eval_placement(params, users, buildings, p_tx_watts, env_key, rng):
    """Returns negative min-user rate (minimising this maximises the bottleneck rate)."""
    x, y, alt = params
    rates = [compute_user_rate((x, y, alt), u, p_tx_watts, env_key, buildings, rng, n_mc=8)
             for u in users]
    return -float(np.min(rates))


def optimize_uav_3d(users, buildings, area_size=500,
                    alt_min=30, alt_max=150,
                    p_tx_watts=None, env_key=DEFAULT_ENV,
                    method="DE"):
    """
    Find the optimal (x, y, altitude) for the UAV using four methods:

    "Grid"  — exhaustive 2D grid at the midpoint altitude. Simple, slow, suboptimal.
    "MC"    — Monte Carlo random search over the full 3D space.
    "DE"    — Differential Evolution (scipy). Population-based global stochastic
               optimiser; no gradients required; handles non-convex objectives well.
               Reference: R. Storn & K. Price, "Differential Evolution — A Simple
               and Efficient Heuristic," J. Global Optimization, vol. 11, 1997.
    "SLSQP" — Sequential Least Squares Programming (gradient-based local search).
               Fast, but may converge to a local optimum.

    Constraints: x, y ∈ [10, area_size−10],  altitude ∈ [alt_min, alt_max].
    Returns (optimal_position, min_rate_achieved, runtime_seconds).
    """
    if p_tx_watts is None:
        p_tx_watts = dbm_to_watts(30)
    rng    = np.random.default_rng(42)
    bounds = [(10, area_size - 10), (10, area_size - 10), (alt_min, alt_max)]
    t0     = time.time()

    if method == "Grid":
        alt_fixed = (alt_min + alt_max) / 2
        xs = np.linspace(10, area_size - 10, 25)
        ys = np.linspace(10, area_size - 10, 25)
        best_val, best_pos = -np.inf, None
        for ux in xs:
            for uy in ys:
                val = -_eval_placement((ux, uy, alt_fixed), users, buildings,
                                       p_tx_watts, env_key, rng)
                if val > best_val:
                    best_val, best_pos = val, (ux, uy, alt_fixed)
        result_pos, result_val = np.array(best_pos), best_val

    elif method == "MC":
        # Uniform random sampling over the full 3D search space
        best_val, best_pos = -np.inf, None
        for _ in range(400):
            x   = rng.uniform(*bounds[0])
            y   = rng.uniform(*bounds[1])
            alt = rng.uniform(*bounds[2])
            val = -_eval_placement((x, y, alt), users, buildings,
                                   p_tx_watts, env_key, rng)
            if val > best_val:
                best_val, best_pos = val, (x, y, alt)
        result_pos, result_val = np.array(best_pos), best_val

    elif method == "DE":
        res = differential_evolution(
            _eval_placement, bounds=bounds,
            args=(users, buildings, p_tx_watts, env_key, rng),
            seed=42, maxiter=80, popsize=8, tol=1e-3,
            mutation=(0.5, 1.0), recombination=0.7,
        )
        result_pos, result_val = res.x, -res.fun

    elif method == "SLSQP":
        x0  = np.array([area_size / 2, area_size / 2, (alt_min + alt_max) / 2])
        res = minimize(_eval_placement, x0,
                       args=(users, buildings, p_tx_watts, env_key, rng),
                       method='SLSQP', bounds=bounds,
                       options={'ftol': 1e-3, 'maxiter': 100})
        result_pos, result_val = res.x, -res.fun
    else:
        raise ValueError(f"Unknown method: {method}")

    return result_pos, result_val, time.time() - t0


def plot_town_simulation():
    """
    Load the real OSM town (or synthetic fallback), then find the optimal UAV
    position using Differential Evolution with full 3D altitude constraints.
    Also runs a grid search for comparison.
    """
    area_size  = 500
    buildings, users, roads = generate_town_osm(area_size=area_size)
    p_tx_watts = dbm_to_watts(30)

    print("  [DE]   Running 3D optimisation (alt 30–150 m) ...")
    pos_de, rate_de, t_de = optimize_uav_3d(
        users, buildings, area_size, alt_min=30, alt_max=150,
        p_tx_watts=p_tx_watts, method="DE")
    print(f"  DE  → ({pos_de[0]:.0f},{pos_de[1]:.0f},{pos_de[2]:.0f}) m  "
          f"min_rate={rate_de:.3f} b/s/Hz  t={t_de:.1f}s")

    print("  [Grid] Running grid search (fixed alt=80 m) ...")
    pos_gs, rate_gs, t_gs = optimize_uav_3d(
        users, buildings, area_size, alt_min=80, alt_max=80,
        p_tx_watts=p_tx_watts, method="Grid")
    print(f"  Grid → ({pos_gs[0]:.0f},{pos_gs[1]:.0f},{pos_gs[2]:.0f}) m  "
          f"min_rate={rate_gs:.3f} b/s/Hz  t={t_gs:.1f}s")

    fig, (ax_map, ax_cmp) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Town map ---
    ax_map.set_facecolor('#1a1a2e')
    for road in roads:
        rx = [p[0] for p in road]; ry = [p[1] for p in road]
        ax_map.plot(rx, ry, color='#f0e68c', lw=1.2, zorder=1)
    for bx, by, bw, bd, bh in buildings:
        ax_map.add_patch(Rectangle((bx - bw/2, by - bd/2), bw, bd,
                                    fc='gray', ec='black', alpha=0.7, zorder=2))
        ax_map.text(bx, by, f"{bh:.0f}m", fontsize=5, ha='center', va='center',
                    color='white', zorder=3)
    ax_map.scatter(users[:, 0], users[:, 1], c='deepskyblue', s=25, zorder=4, label='Users')
    ax_map.plot(*pos_de[:2], 'r*', ms=18, zorder=6, label=f"DE  alt={pos_de[2]:.0f}m")
    ax_map.plot(*pos_gs[:2], 'g^', ms=13, zorder=6, label=f"Grid alt={pos_gs[2]:.0f}m")
    for u in users:
        los = has_line_of_sight(tuple(pos_de), u, buildings)
        ax_map.plot([pos_de[0], u[0]], [pos_de[1], u[1]],
                    color='lime' if los else 'red',
                    ls='-' if los else '--', alpha=0.2, lw=0.7)
    ax_map.set_xlim(0, area_size); ax_map.set_ylim(0, area_size)
    ax_map.set_xlabel("X (m)"); ax_map.set_ylabel("Y (m)")
    ax_map.set_title("Town Map — Optimal UAV Positions\n(green=LoS, red=NLoS from DE solution)")
    ax_map.legend(fontsize=8); ax_map.set_aspect('equal')

    # --- Method comparison bar chart ---
    labels = ['Grid Search\n(2D, fixed alt)', 'Diff. Evolution\n(3D, global)']
    rates  = [rate_gs, rate_de]
    alts   = [pos_gs[2], pos_de[2]]
    times  = [t_gs, t_de]
    xb     = np.arange(2)
    ax_t   = ax_cmp.twinx()
    ax_cmp.bar(xb - 0.2, rates, 0.35, color=['steelblue', 'coral'],
               label='Min User Rate (b/s/Hz)')
    ax_t.bar(xb + 0.2, times, 0.35, color=['steelblue', 'coral'], alpha=0.35,
             label='Runtime (s)')
    for i, (r, a) in enumerate(zip(rates, alts)):
        ax_cmp.text(i - 0.2, r + 0.005, f"{r:.3f}\nalt={a:.0f}m", ha='center', fontsize=9)
    ax_cmp.set_xticks(xb); ax_cmp.set_xticklabels(labels)
    ax_cmp.set_ylabel("Min User Rate (bits/s/Hz)", color='steelblue')
    ax_t.set_ylabel("Runtime (s)", color='gray')
    ax_cmp.set_title("Optimisation Method Comparison")
    ax_cmp.grid(axis='y', alpha=0.3)
    fig.suptitle("UAV Placement Optimisation on Real Town Layout", fontsize=13)
    plt.tight_layout()


# =============================================================
# SECTION 6: Fair Resource Allocation (full channel model)
# =============================================================

def plot_resource_allocation():
    """
    Four bandwidth allocation strategies evaluated with the full Al-Hourani model.
    Users are at increasing distances; the full channel model (path loss + shadowing
    + fading) replaces the simple FSPL used in v1.
    """
    rng        = np.random.default_rng(99)
    distances  = np.array([50, 100, 150, 200, 300, 400, 500, 700], dtype=float)
    num_users  = len(distances)
    uav_pos    = (0, 0, 100)
    p_tx_watts = dbm_to_watts(30)

    # Ergodic capacity per user with full channel model
    capacity_per_hz = np.array([
        compute_user_rate(uav_pos, (d, 0, 0), p_tx_watts,
                          buildings=None, rng=rng, n_mc=30)
        for d in distances
    ])
    total_bw = 1.0

    # Strategy 1: Max-Throughput — all bandwidth to the best user
    w_greedy        = np.zeros(num_users)
    w_greedy[np.argmax(capacity_per_hz)] = total_bw
    rate_greedy     = w_greedy * capacity_per_hz

    # Strategy 2: Equal bandwidth split
    rate_equal      = np.full(num_users, total_bw / num_users) * capacity_per_hz

    # Strategy 3: Max-Min — equalize all rates; solved analytically
    R_maxmin        = total_bw / np.sum(1.0 / capacity_per_hz)
    rate_maxmin     = np.full(num_users, R_maxmin)

    # Strategy 4: Proportional Fairness — maximize Σ log(rate_i), solved via SLSQP
    def neg_pf(w):
        return -np.sum(np.log(np.maximum(w * capacity_per_hz, 1e-15)))
    res     = minimize(neg_pf, np.full(num_users, total_bw / num_users),
                       method='SLSQP',
                       bounds=[(1e-6, total_bw)] * num_users,
                       constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - total_bw})
    rate_pf = res.x * capacity_per_hz

    def jains_index(rates):
        return np.sum(rates)**2 / (num_users * np.sum(rates**2))

    schemes = {"Max-Throughput": rate_greedy, "Equal BW": rate_equal,
               "Proportional": rate_pf, "Max-Min": rate_maxmin}
    colors  = ["tab:red", "tab:blue", "tab:orange", "tab:green"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(num_users)
    for idx, (name, rates) in enumerate(schemes.items()):
        ax1.bar(x + idx * 0.2, rates, 0.2, label=name, color=colors[idx])
    ax1.set_xlabel("User Index")
    ax1.set_ylabel("Throughput (bits/s/Hz)")
    ax1.set_title("Per-User Throughput — Full Al-Hourani Channel Model")
    ax1.set_xticks(x + 0.3)
    ax1.set_xticklabels([f"U{i}\n({d:.0f}m)" for i, d in enumerate(distances)], fontsize=7)
    ax1.legend(fontsize=8); ax1.grid(axis='y', alpha=0.3)

    fairness = [jains_index(r) for r in schemes.values()]
    sum_r    = [np.sum(r)       for r in schemes.values()]
    ax2b = ax2.twinx()
    ax2.bar(np.arange(4) - 0.15, fairness, 0.3, color='steelblue', label="Jain's Fairness")
    ax2b.bar(np.arange(4) + 0.15, sum_r,   0.3, color='coral',     label="Sum Rate")
    ax2.set_ylabel("Jain's Fairness Index (0–1)", color='steelblue')
    ax2b.set_ylabel("Sum Rate (bits/s/Hz)", color='coral')
    ax2.set_xticks(np.arange(4)); ax2.set_xticklabels(list(schemes.keys()), fontsize=8)
    ax2.set_title("Fairness vs Total Throughput Tradeoff")
    ax2.set_ylim(0, 1.1); ax2.grid(axis='y', alpha=0.3)
    fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=8)
    plt.tight_layout()


# =============================================================
# SECTION 7: UAV-to-Vehicle Communication & Trajectory
# =============================================================

def generate_vehicles(num_vehicles=10, area_size=500, seed=77):
    """
    Vehicles driving on a simplified grid road network.
    Each vehicle has a data buffer, priority (1=normal, 3=emergency), and QoS requirement.
    """
    rng = np.random.default_rng(seed)
    vehicles = []
    for vid in range(num_vehicles):
        if vid % 2 == 0:   # East-West lane
            x  = rng.uniform(0, area_size)
            y  = rng.choice([100, 200, 300, 400]) + rng.normal(0, 5)
            vx = rng.uniform(8, 22) * rng.choice([-1, 1])
            vy = rng.normal(0, 0.5)
        else:               # North-South lane
            x  = rng.choice([100, 200, 300, 400]) + rng.normal(0, 5)
            y  = rng.uniform(0, area_size)
            vx = rng.normal(0, 0.5)
            vy = rng.uniform(8, 22) * rng.choice([-1, 1])
        vehicles.append({
            'id':                vid,
            'position':          np.array([x, y], dtype=float),
            'velocity':          np.array([vx, vy], dtype=float),
            'speed_kmh':         np.sqrt(vx**2 + vy**2) * 3.6,
            'channel_quality_dB': rng.uniform(5, 25),
            'buffer_bytes':      rng.uniform(1e3, 1e6),
            'priority':          rng.choice([1, 1, 1, 2, 3]),
            'qos_latency_ms':    rng.choice([50, 100, 200]),
        })
    return vehicles


def simulate_uav_trajectory(vehicles, area_size=500, T=60, dt=1.0,
                              uav_alt=80, uav_max_speed=20):
    """
    Greedy weighted-centroid trajectory: each second the UAV moves toward the
    weighted average position of all vehicles, where the weight favours vehicles
    with more buffered data, higher priority, and closer proximity.
    """
    uav_pos            = np.array([area_size / 2, area_size / 2], dtype=float)
    uav_trajectory     = [uav_pos.copy()]
    vehicle_trajectories = {v['id']: [v['position'].copy()] for v in vehicles}
    served_data        = {v['id']: [0.0] for v in vehicles}

    for _ in np.arange(0, T, dt)[1:]:
        for v in vehicles:
            v['position'] = (v['position'] + v['velocity'] * dt) % area_size
            vehicle_trajectories[v['id']].append(v['position'].copy())

        positions = np.array([v['position'] for v in vehicles])
        weights   = np.array([v['buffer_bytes'] * v['priority'] for v in vehicles])
        dists     = np.linalg.norm(positions - uav_pos, axis=1)
        weights   = weights / (dists + 10)
        weights  /= weights.sum()

        direction = np.sum(positions * weights[:, None], axis=0) - uav_pos
        step      = np.linalg.norm(direction)
        if step > uav_max_speed * dt:
            direction = direction / step * uav_max_speed * dt
        uav_pos = uav_pos + direction
        uav_trajectory.append(uav_pos.copy())

        for v in vehicles:
            d3d   = np.sqrt(np.sum((v['position'] - uav_pos)**2) + uav_alt**2)
            snr   = dbm_to_watts(30) / (10 ** (fspl_dB(d3d) / 10)) / noise_power_watts
            rate  = channel_bandwidth_hz * np.log2(1 + snr)
            bytes_served = rate * (1.0 / len(vehicles)) * dt / 8
            served_data[v['id']].append(served_data[v['id']][-1] + bytes_served)

    return np.array(uav_trajectory), vehicle_trajectories, served_data


def plot_vehicle_communication():
    """Trajectory map, cumulative served data, and vehicle status dashboard."""
    area_size = 500
    vehicles  = generate_vehicles(area_size=area_size)
    uav_traj, veh_trajs, served_data = simulate_uav_trajectory(
        vehicles, area_size=area_size)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.cm.tab10

    ax = axes[0]
    for v in vehicles:
        traj = np.array(veh_trajs[v['id']])
        c    = cmap(v['id'] % 10)
        ax.plot(traj[:, 0], traj[:, 1], '-', color=c, alpha=0.5, lw=1)
        ax.plot(traj[0, 0],  traj[0, 1],  'o', color=c, ms=5)
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=c, ms=5)
        ax.annotate(f"V{v['id']}", traj[0], fontsize=6)
    ax.plot(uav_traj[:, 0], uav_traj[:, 1], 'k-',  lw=2.5, label='UAV path')
    ax.plot(uav_traj[0, 0],  uav_traj[0, 1],  'r^', ms=12,  label='UAV start')
    ax.plot(uav_traj[-1, 0], uav_traj[-1, 1], 'r*', ms=14,  label='UAV end')
    for i in range(0, len(uav_traj), 10):
        ax.annotate(f"{i}s", uav_traj[i], fontsize=6, color='red')
    ax.set_xlim(0, area_size); ax.set_ylim(0, area_size)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("UAV & Vehicle Trajectories")
    ax.legend(fontsize=7); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

    ax = axes[1]
    for v in vehicles:
        data = np.array(served_data[v['id']])
        ax.plot(np.linspace(0, 60, len(data)), data / 1e3,
                label=f"V{v['id']} (p={v['priority']})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cumulative Data Served (KB)")
    ax.set_title("Data Served to Each Vehicle")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[2]; ax.axis('off')
    col_labels = ["ID", "Pos (x,y)", "Speed\n(km/h)", "Buffer\n(KB)",
                  "Priority", "QoS\n(ms)", "SNR\n(dB)"]
    tbl = ax.table(
        cellText=[[f"V{v['id']}",
                   f"({veh_trajs[v['id']][30][0]:.0f},{veh_trajs[v['id']][30][1]:.0f})",
                   f"{v['speed_kmh']:.1f}", f"{v['buffer_bytes']/1e3:.1f}",
                   f"{v['priority']}", f"{v['qos_latency_ms']}",
                   f"{v['channel_quality_dB']:.1f}"] for v in vehicles],
        colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1.0, 1.3)
    ax.set_title("Vehicle Reports at t = 30 s", fontsize=10, pad=20)
    plt.tight_layout()


# =============================================================
# SECTION 8: Energy Efficiency Analysis
# =============================================================

def plot_energy_efficiency():
    """
    Three-panel EE analysis:
    1. EE vs Tx Power — for each modulation, shows the optimal operating point.
    2. EE vs UAV Altitude — how height affects the energy cost per bit.
    3. EE vs Capacity Pareto curve — the fundamental speed vs. efficiency tradeoff.
    """
    rng        = np.random.default_rng(0)
    ref_user   = (200, 0, 0)
    uav_base   = (0, 0, 100)
    p_range_dBm = np.linspace(-10, 40, 80)
    p_range_w   = dbm_to_watts(p_range_dBm)
    altitudes   = np.linspace(20, 200, 60)
    p_fixed_w   = dbm_to_watts(23)    # 200 mW fixed Tx power
    colors      = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: EE vs Tx Power ---
    ax = axes[0]
    for (mod_name, M), col in zip(MODULATIONS.items(), colors):
        ee_vals = []
        for ptx in p_range_w:
            snr = compute_snr_full(uav_base, ref_user, ptx, apply_fading=False, rng=rng)
            cap = np.log2(1 + max(snr, 0))
            ee_vals.append(energy_efficiency(cap, ptx))
        ax.plot(p_range_dBm, ee_vals, color=col, lw=2, label=mod_name)
        pk = int(np.argmax(ee_vals))
        ax.plot(p_range_dBm[pk], ee_vals[pk], 'o', color=col, ms=8)
    ax.set_xlabel("Tx Power (dBm)")
    ax.set_ylabel("Energy Efficiency (bits/J/Hz)")
    ax.set_title("EE vs Tx Power\n(● = optimal operating point)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Panel 2: EE vs UAV Altitude ---
    ax = axes[1]
    for (mod_name, M), col in zip(MODULATIONS.items(), colors):
        ee_alt = []
        for alt in altitudes:
            snr = compute_snr_full((0, 0, alt), ref_user, p_fixed_w,
                                   apply_fading=False, rng=rng)
            cap = np.log2(1 + max(snr, 0))
            ee_alt.append(energy_efficiency(cap, p_fixed_w))
        ax.plot(altitudes, ee_alt, color=col, lw=2, label=mod_name)
    ax.set_xlabel("UAV Altitude (m)")
    ax.set_ylabel("Energy Efficiency (bits/J/Hz)")
    ax.set_title(f"EE vs Altitude\n(Ptx = {watts_to_dbm(p_fixed_w):.0f} dBm, user at 200 m)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Panel 3: EE vs Capacity Pareto curve ---
    ax = axes[2]
    for (mod_name, M), col in zip(MODULATIONS.items(), colors):
        caps, ees = [], []
        for ptx in p_range_w:
            snr = compute_snr_full(uav_base, ref_user, ptx, apply_fading=False, rng=rng)
            cap = np.log2(1 + max(snr, 0))
            caps.append(cap)
            ees.append(energy_efficiency(cap, ptx))
        ax.plot(caps, ees, color=col, lw=2, label=mod_name)
    ax.set_xlabel("Capacity (bits/s/Hz)")
    ax.set_ylabel("Energy Efficiency (bits/J/Hz)")
    ax.set_title("EE vs Capacity (Pareto)\n(sweep Tx power; rightward = more power)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Energy Efficiency Analysis — Al-Hourani Channel + Dipole Antenna",
                 fontsize=13)
    plt.tight_layout()


# =============================================================
# SECTION 9: Optimization Technique Comparison
# =============================================================

def plot_optimization_comparison():
    """
    Benchmark four UAV placement optimisation algorithms side-by-side:
    Grid Search, Monte Carlo, Differential Evolution, and SLSQP.

    Metrics reported:
    - Achieved minimum user rate (solution quality)
    - Optimal altitude discovered
    - Wall-clock runtime
    """
    area_size  = 500
    buildings, users, _ = generate_town(area_size=area_size,
                                         num_buildings=10, num_users=15, seed=42)
    p_tx_watts = dbm_to_watts(30)
    methods    = ["Grid", "MC", "DE", "SLSQP"]
    labels     = ["Grid Search\n(2D, fixed alt)", "Monte Carlo\n(random 3D)",
                  "Diff. Evolution\n(3D, global)", "SLSQP\n(gradient, 3D)"]
    results    = {}

    print("  Benchmarking optimisation methods (expect ~60–120 s total) ...")
    for m in methods:
        print(f"    {m} ...", end=' ', flush=True)
        pos, rate, elapsed = optimize_uav_3d(
            users, buildings, area_size, alt_min=30, alt_max=150,
            p_tx_watts=p_tx_watts, method=m)
        results[m] = {"pos": pos, "rate": rate, "time": elapsed}
        print(f"rate={rate:.3f} b/s/Hz  alt={pos[2]:.0f}m  t={elapsed:.1f}s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x      = np.arange(len(methods))
    bcolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    axes[0].bar(x, [results[m]['rate'] for m in methods], color=bcolors)
    for i, m in enumerate(methods):
        axes[0].text(i, results[m]['rate'] + 0.003,
                     f"{results[m]['rate']:.3f}", ha='center', fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Min User Rate (bits/s/Hz)")
    axes[0].set_title("Solution Quality  (higher = better)")
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x, [results[m]['pos'][2] for m in methods], color=bcolors)
    for i, m in enumerate(methods):
        axes[1].text(i, results[m]['pos'][2] + 1,
                     f"{results[m]['pos'][2]:.0f} m", ha='center', fontsize=9)
    axes[1].axhline(30,  color='gray', ls='--', lw=0.8, label="alt_min = 30 m")
    axes[1].axhline(150, color='gray', ls=':',  lw=0.8, label="alt_max = 150 m")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Optimal Altitude Found (m)")
    axes[1].set_title("Altitude Selected by Each Method")
    axes[1].legend(fontsize=8); axes[1].grid(axis='y', alpha=0.3)

    axes[2].bar(x, [results[m]['time'] for m in methods], color=bcolors)
    for i, m in enumerate(methods):
        axes[2].text(i, results[m]['time'] + 0.2,
                     f"{results[m]['time']:.1f}s", ha='center', fontsize=9)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("Runtime (seconds)")
    axes[2].set_title("Computation Time  (lower = faster)")
    axes[2].grid(axis='y', alpha=0.3)

    fig.suptitle("Optimisation Method Comparison: Grid vs Monte Carlo vs DE vs SLSQP",
                 fontsize=13)
    plt.tight_layout()


# =============================================================
# MAIN
# =============================================================

def main():
    print("UAV Extended Simulation v2")
    print("=" * 65)

    print("\n[1/9] BER vs SNR — multiple modulations ...")
    plot_ber_vs_snr()

    print("[2/9] Modulation comparison + energy efficiency ...")
    plot_modulation_comparison(target_ber=1e-4)

    print("[3/9] 3D coordinate geometry ...")
    plot_3d_geometry()

    print("[4/9] Antenna pattern comparison (narrow cosine vs dipole) ...")
    plot_antenna_pattern()

    print("[5/9] Real town layout + constrained 3D optimisation ...")
    plot_town_simulation()

    print("[6/9] Fair resource allocation (full channel model) ...")
    plot_resource_allocation()

    print("[7/9] UAV-to-vehicle communication & trajectory ...")
    plot_vehicle_communication()

    print("[8/9] Energy efficiency analysis ...")
    plot_energy_efficiency()

    print("[9/9] Optimisation technique comparison ...")
    plot_optimization_comparison()

    print("\nAll plots generated. Close windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
