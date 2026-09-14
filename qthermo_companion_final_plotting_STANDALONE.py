from __future__ import annotations

"""
Final publication plotting and cycle-audit script for the QThermodynamics
companion paper.

Purpose
-------
The publication figures still use the finalized numerical values embedded in
this file, so no external result folders are required.  In addition, this
revision implements the constructive cycle described in the revised manuscript
and keeps separate the three quantum descriptions used there:

    rho              : exact microscopic state,
    bar(rho)_R        : record-only maximum-entropy representative G_R[rho],
    bar(rho)_{E,R}    : energy-and-record maximum-entropy representative.

The record-only split is the linear retained/unresolved decomposition

    rho = bar(rho)_R + chi_R,

whereas the energy-refined description gives

    rho = bar(rho)_{E,R} + chi_{E,R}
        = bar(rho)_R + delta_{E|R} + chi_{E,R}.

Thermodynamic cycle used by the manuscript
------------------------------------------
The physically explicit cycle is

    A -> B -> C^- -> C^+ -> Cbar -> D -> A.

    A -> B       : reversible hot isotherm at T_h;
    B -> C^-     : actual thermally isolated finite-time work stroke;
    C^- -> C^+   : work-only switch H_0 -> H_{C,R}; microscopic state unchanged;
    C^+ -> Cbar  : fixed-H_{C,R} relaxation to bar(rho)_{E,R}(C) at T_c;
                   W = 0 and net Q = 0 by the endpoint energy identity;
    Cbar -> D    : genuine reversible cold isotherm at T_c;
    D -> A       : reversible thermally isolated Hamiltonian scaling.

The record-relative irreversibility created during B -> C^- is

    Sigma_BC = S_{E,R}(C) - S_{E,R}(B).

The microscopic entropy increase during C^+ -> Cbar equals this same number.
It is the physical realization of the already identified inaccessible
information, not a second independent contribution.  In the retained (E,R)
thermodynamic balance, C^+ and Cbar are represented by the same state and the
interface has zero net heat.  Therefore

    Sigma_cyc = -Q_h/T_h + Q_c/T_c = Sigma_BC.

Two different audits are provided below:

1. A fast retained-state ledger based on the finalized embedded publication
   numbers for both quantum and classical cycles.
2. An exact constructive Q=4 quantum audit that rebuilds the Bose-Hubbard
   Hamiltonian, propagates the actual rho_C, constructs bar(rho)_R and
   bar(rho)_{E,R}, installs H_{C,R}, verifies the zero-net-heat interface, and
   independently thermodynamically integrates the hot and cold reversible
   paths.  This audit is enabled by default and is fast.

An optional classical microscopic audit is also included.  It regenerates the
natural classical endpoint using scrambled Sobol sampling and propagates the
DNLS ensemble before constructing the cold-side closure.  It is deliberately
disabled by default because it is much more expensive than figure generation;
set RUN_FULL_CLASSICAL_CYCLE_AUDIT = True when a fresh classical closure audit
is wanted.

Protocol-duration robustness
----------------------------
Changing tau_BC changes the actual endpoint C, both thermodynamic
representatives, and the record-controlled Hamiltonian H_{C,R}(tau_BC).  The
embedded residuals in Fig. 4(d) audit the endpoint-specific (E,R)
reconstruction.  They do not by themselves simulate the cold relaxation.
The physical continuation for each endpoint is the constructive sequence
C^- -> C^+ -> Cbar -> D described above.

Main figures
------------
Fig1_framework_and_geometry.pdf
Fig2_energy_resolved_mechanism.pdf
Fig3_quantum_classical_crossover.pdf
Fig4_robustness.pdf

Supplementary figures
---------------------
FigS1_matched_preparation_validation.pdf
FigS2_numerical_refinement.pdf

Run in Jupyter with, for example:
    %run qthermo_companion_final_plotting_CONSTRUCTIVE_CYCLE.py

The figures are written to
    companion_paper_figures_final/
under the current Jupyter working directory.
"""

from pathlib import Path
from dataclasses import dataclass
from itertools import product
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.ticker import MaxNLocator

from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm
from scipy.optimize import minimize, brentq
from scipy.special import i0, i1, ndtri
from scipy.stats import qmc


# =============================================================================
# Paths and run controls
# =============================================================================

# Self-contained: no external result folders are needed.
OUTDIR = Path.cwd() / "companion_paper_figures_final"
OUTDIR.mkdir(parents=True, exist_ok=True)

MAKE_MAIN_FIGURES = True
MAKE_SUPPLEMENTARY = True

# Fast exact audit of the representative Q=4 quantum cycle.
RUN_EXACT_QUANTUM_CYCLE_AUDIT = True

# Full classical rerun is much more expensive than plotting.  When enabled,
# these are the production-form settings used by the natural thermal audit.
RUN_FULL_CLASSICAL_CYCLE_AUDIT = False
CLASSICAL_AUDIT_QMC_POWER = 16
CLASSICAL_AUDIT_SCRAMBLES = 8
CLASSICAL_AUDIT_DT = 0.05
CLASSICAL_AUDIT_QUADRATURE_ORDER = 16


# =============================================================================
# User's publication plot style
# =============================================================================

MAIN_TEXT_FONT_SIZE = 14

NPG = [
    "#E64B35",  # red
    "#4DBBD5",  # cyan
    "#00A087",  # green
    "#3C5488",  # navy
    "#F39B7F",  # salmon
    "#8491B4",  # lavender
    "#91D1C2",  # mint
    "#DC0000",  # deep red
    "#7E6148",  # brown
]

RED = NPG[0]
CYAN = NPG[1]
GREEN = NPG[2]
NAVY = NPG[3]
SALMON = NPG[4]
LAVENDER = NPG[5]
MINT = NPG[6]
DEEP_RED = NPG[7]
BROWN = NPG[8]
BLACK = "black"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D9D9D9"

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
MAIN_FONT = "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"

plt.rcParams.update(
    {
        "font.family": MAIN_FONT,
        "font.size": MAIN_TEXT_FONT_SIZE,
        "axes.labelsize": MAIN_TEXT_FONT_SIZE,
        "xtick.labelsize": MAIN_TEXT_FONT_SIZE,
        "ytick.labelsize": MAIN_TEXT_FONT_SIZE,
        "legend.fontsize": MAIN_TEXT_FONT_SIZE,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "legend.frameon": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.max_open_warning": 0,
        "axes.unicode_minus": True,
    }
)


def finish_axes(ax, minor=True):
    """Nature-style fully bounded axes with inward ticks on all sides."""
    if minor:
        ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)


def panel_label(ax, label, x=0.03, y=0.95):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=BLACK,
        zorder=20,
    )


def savefig(fig, filename: str, h_pad=None, w_pad=None):
    """Save as tightly cropped PDF, show, and intentionally do not close."""
    path = OUTDIR / filename
    fig.tight_layout(pad=0.35, h_pad=h_pad, w_pad=w_pad)
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=False,
    )
    print(f"Saved: {path}")
    plt.show()
    return path


def exact_pi_L4_Q4():
    """Two halves of two sites each, Q=4 bosons: Omega_r=(r+1)(Q-r+1)."""
    Q = 4
    r = np.arange(Q + 1)
    omega = (r + 1) * (Q - r + 1)
    return omega / omega.sum()


def relative_entropy_discrete(p, pi):
    p = np.asarray(p, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / pi[mask])))


# =============================================================================
# Final validated numerical data embedded in this script
# =============================================================================

# Particle-number scaling, matched-pi comparison (L=4).
qscale = pd.DataFrame({
    "Q": [2, 3, 4, 5, 6, 7],
    "sigma_q": [0.40759643, 0.546515758, 0.6416098261868568, 0.6996893087, 0.7348307439, 0.7566884905797551],
    "sigma_cl": [0.7805010187, 0.7802359454, 0.7789281846, 0.7785369564, 0.7783656217, 0.7763376307518322],
    "se_cl": [0.0031808589, 0.0031347387, 0.0032983204, 0.0031809227, 0.0031606203, 0.003224693466196173],
    "gap": [0.3729045887, 0.2337201874, 0.13731835843661333, 0.0788476477, 0.0435348778, 0.019649140172077084],
    "reduction_pct": [47.77759154255924, 29.955065359130533, 17.629142345515756, 10.127669220045258, 5.593114159502194, 2.5310044745671],
    "reduction_pct_se": [0.21282754120931385, 0.28141816414922655, 0.3487940052167823, 0.3671976451853722, 0.383347249067308, 0.40485945879395435],
    "eta_adv_pp": [14.528806, 7.414235, 3.929182, 2.106958, 1.10933, 0.4844818242802848],
    "eta_adv_pp_se": [0.12393, 0.099442, 0.094377, 0.085, 0.080537, 0.07951011390654086],
})

# -----------------------------------------------------------------------------
# Constructive complete-cycle bookkeeping
# -----------------------------------------------------------------------------
#
# The physical cold-side closure is
#
#   C^- -> C^+ -> Cbar -> D.
#
# C^- -> C^+ is a work-only Hamiltonian switch.  C^+ -> Cbar is a
# fixed-H_{C,R} relaxation with zero net heat; microscopically its entropy
# increase equals the already identified record-relative Sigma_BC.  Cbar -> D
# is the genuinely reversible cold isotherm.  Thus the retained complete-cycle
# entropy balance still gives Sigma_cyc = Sigma_BC, but we do not represent the
# whole cold side by a fictitious single "Sigma_CD=0" stroke.
#
# Q_h is fixed by the equilibrium A and B endpoints.  The values below are the
# finalized publication values.

CYCLE_CLOSURE_LABEL = (
    "constructive C^- -> C^+ -> Cbar -> D closure plus reversible D->A scaling"
)

_QH_QSCALE = np.array([
    1.924992656490392,
    3.5463567493615438,
    5.242260762756452,
    7.016720371698729,
    8.829965924866576,
    10.646218365018088,
], dtype=float)

qscale["Q_h"] = _QH_QSCALE
qscale["T_c"] = qscale["Q"].to_numpy(dtype=float) * 0.375
qscale["sigma_BC_q"] = qscale["sigma_q"]
qscale["sigma_BC_cl"] = qscale["sigma_cl"]
qscale["sigma_cyc_q"] = qscale["sigma_BC_q"]
qscale["sigma_cyc_cl"] = qscale["sigma_BC_cl"]
qscale["sigma_cyc_gap"] = qscale["sigma_cyc_cl"] - qscale["sigma_cyc_q"]
qscale["sigma_cyc_gap_se"] = qscale["se_cl"]
qscale["eta_adv_pp"] = (
    100.0 * qscale["T_c"] * qscale["sigma_cyc_gap"] / qscale["Q_h"]
)
qscale["eta_adv_pp_se"] = (
    100.0 * qscale["T_c"] * qscale["sigma_cyc_gap_se"] / qscale["Q_h"]
)
qscale["cycle_reduction_pct"] = (
    100.0 * qscale["sigma_cyc_gap"] / qscale["sigma_cyc_cl"]
)
qscale["cycle_reduction_pct_se"] = (
    100.0
    * qscale["sigma_cyc_q"]
    * qscale["sigma_cyc_gap_se"]
    / (qscale["sigma_cyc_cl"] ** 2)
)

# Exact mechanism decomposition at Q=4.
mechanism = pd.DataFrame([
    {"case":"thermal_quantum", "Sigma_BC":0.6416098261868568, "Sigma_BC_se":0.0, "DeltaG_R":1.7377442646484222, "DeltaG_R_se":0.0, "DeltaI_E":1.0961344386992655, "DeltaI_E_se":0.0},
    {"case":"thermal_classical", "Sigma_BC":0.7789281846234701, "Sigma_BC_se":0.0032983204136428817, "DeltaG_R":1.5697129822787093, "DeltaG_R_se":0.0004363631028507616, "DeltaI_E":0.790784797655239, "DeltaI_E_se":0.0031801900681991935},
    {"case":"matched_pE_quantum", "Sigma_BC":0.6416097957567255, "Sigma_BC_se":0.0, "DeltaG_R":1.737744264648424, "DeltaG_R_se":0.0, "DeltaI_E":1.0961344688916985, "DeltaI_E_se":0.0},
    {"case":"matched_pE_classical", "Sigma_BC":1.9324886995987747, "Sigma_BC_se":0.0022347218332626376, "DeltaG_R":1.5052542322167692, "DeltaG_R_se":0.00034626642970324545, "DeltaI_E":-0.4272344673820002, "DeltaI_E_se":0.002287646976763362},
])

# Interaction-strength robustness.
interaction = pd.DataFrame({
    "g": [0.0, 0.3, 0.7, 1.3, 2.0],
    "sigma_q": [0.7823487486278533, 0.7496477806427573, 0.7029474951380674, 0.6416098259491567, 0.6018646974418442],
    "sigma_cl": [2.2476239852212094, 2.1504790717816338, 2.096893605996989, 1.942141249627961, 1.7991315194772584],
    "sigma_cl_se": [0.005035236626990422, 0.010175634188462132, 0.00465966984341542, 0.0025547659416151713, 0.012191876813613304],
})

# Protocol-duration robustness at L=4, Q=4.
protocol = pd.DataFrame({
    "tau_exp_Jinv": [5.0, 10.0, 20.0, 40.0, 80.0],
    "sigma_BC_quantum": [0.9193084224353518, 0.9660481351994923, 0.6416098261868568, 0.37127197628474606, 0.3090973020390759],
    "sigma_BC_classical_mean": [1.058532281681991, 1.138989863272217, 0.7789281846234702, 0.40302943335106167, 0.33343855359411845],
    "sigma_BC_classical_se": [0.0016520102901411557, 0.002554040792865537, 0.0032983204136428817, 0.002978984488473411, 0.002030461789956714],
    "gap_cl_minus_q": [0.13922385924663927, 0.1729417280727248, 0.13731835843661344, 0.03175745706631561, 0.02434125155504252],
    "same_Qh_delta_eta_identity": [0.03983697078818151, 0.04948486995383353, 0.03929173823596947, 0.009086954608954947, 0.006964910557666602],
    "same_Qh_delta_eta_identity_se": [0.00047269976587520216, 0.000730803247430195, 0.0009437685083530393, 0.0008523949751710807, 0.0005809883984736397],
    "pC_quantum_0": [0.43785621171702216, 0.25343540769770523, 0.1290674769151224, 0.13887231974636988, 0.10857579568723993],
    "pC_quantum_1": [0.3450838481010148, 0.32684737842058437, 0.29121597017170847, 0.25174467679685847, 0.24891824868895074],
    "pC_quantum_2": [0.16963837567639417, 0.25740729985676425, 0.2993206291404099, 0.2869644487015307, 0.3035937576022488],
    "pC_quantum_3": [0.04346788753370326, 0.1339769337228588, 0.19013271907530957, 0.2327088983030462, 0.24672507915717187],
    "pC_quantum_4": [0.003953676971865661, 0.028332980302087484, 0.09026320469744965, 0.08970965645219453, 0.09218711886438861],
    # Each duration produces a different endpoint C and is fitted independently.
    # These residuals are the maximum constraint mismatch in the C-specific
    # energy-and-record representative used to define H_{C,R}(tau).
    "quantum_ER_residual": [5.170521e-7, 4.913915e-7, 4.498258e-7, 3.408335e-8, 4.644712e-8],
    "classical_ER_max_residual": [3.497203e-15, 2.331468e-15, 2.220446e-15, 1.665335e-15, 1.276756e-15],
})

# The protocol-duration scan changes only B -> C^-.  A, B, D, and the
# reversible D -> A scaling are fixed.  Every changed C has its own
# bar(rho)_{E,R}(C) and H_{C,R}(tau_BC).  The subsequent physical closure is
# C^- -> C^+ -> Cbar -> D.  The C^+ -> Cbar interface has zero net heat and
# leaves the retained (E,R) state unchanged; Cbar -> D is quasistatic.
_PROTOCOL_QH = 5.242260762756452
_PROTOCOL_TC = 1.5

protocol["Q_h"] = _PROTOCOL_QH
protocol["T_c"] = _PROTOCOL_TC
protocol["sigma_BC_q"] = protocol["sigma_BC_quantum"]
protocol["sigma_BC_cl"] = protocol["sigma_BC_classical_mean"]
protocol["sigma_cyc_q"] = protocol["sigma_BC_q"]
protocol["sigma_cyc_cl"] = protocol["sigma_BC_cl"]
protocol["sigma_cyc_gap"] = protocol["sigma_cyc_cl"] - protocol["sigma_cyc_q"]
protocol["sigma_cyc_gap_se"] = protocol["sigma_BC_classical_se"]
protocol["same_Qh_delta_eta_identity"] = (
    protocol["T_c"] * protocol["sigma_cyc_gap"] / protocol["Q_h"]
)
protocol["same_Qh_delta_eta_identity_se"] = (
    protocol["T_c"] * protocol["sigma_cyc_gap_se"] / protocol["Q_h"]
)
protocol["cycle_reduction_pct"] = (
    100.0 * protocol["sigma_cyc_gap"] / protocol["sigma_cyc_cl"]
)

# L=6 spot checks.
l6 = pd.DataFrame({
    "Q": [3, 4],
    "sigma_BC_quantum": [0.9451595614396666, 1.1348398013221845],
    "sigma_BC_classical_mean": [1.2592955148654914, 1.2552641513728822],
    "sigma_BC_classical_se": [0.002592545302965668, 0.002651324710610031],
})

# Final strict matched-(p_B,E_B) preparation summary.
matched_final = pd.DataFrame([
    {"quantity":"sigma_BC_quantum", "value":0.6416098261868568, "standard_error":0.0},
    {"quantity":"sigma_BC_classical_matched_preparation", "value":1.9324886995987747, "standard_error":0.0022347218332626376},
    {"quantity":"quantum_reduction_percent_relative_to_classical", "value":66.79877991938227, "standard_error":0.0},
])

# Target-adapted matched-preparation convergence and endpoint p_C.
matched_conv = pd.DataFrame([
    {"power":14, "n_per_bin":16384, "sigma_BC_mean":1.9364444671578647, "sigma_BC_se":0.0029487937010267266,
     "pC_0_mean":0.10155716183434949, "pC_0_se":0.0006774400877350119,
     "pC_1_mean":0.4401076374471775, "pC_1_se":0.001762348320600787,
     "pC_2_mean":0.3558729282976114, "pC_2_se":0.0022040187736061994,
     "pC_3_mean":0.09122861036858791, "pC_3_se":0.0005308218833110142,
     "pC_4_mean":0.0112336620522736, "pC_4_se":0.00016479872867433806},
    {"power":16, "n_per_bin":65536, "sigma_BC_mean":1.9324886995987747, "sigma_BC_se":0.0022347218332626376,
     "pC_0_mean":0.10085546130887166, "pC_0_se":0.00035470389236909783,
     "pC_1_mean":0.44287530718349566, "pC_1_se":0.00039294137600329746,
     "pC_2_mean":0.354532114073271, "pC_2_se":0.0004986347791754013,
     "pC_3_mean":0.09028587340898489, "pC_3_se":0.00038764487911554715,
     "pC_4_mean":0.011451244025376825, "pC_4_se":0.00007961425188482628},
])

# Deterministic maximum-entropy quadrature convergence.
quad_conv = pd.DataFrame({
    "order": [8, 10, 12, 16, 20, 24, 32, 40],
    "S_B": [-6.3801742194326145, -6.38054246918717, -6.380544721814977, -6.3805447275476554, -6.380544727547657, -6.380544727547663, -6.380544727547657, -6.380544727547662],
    "S_ER_C_representative": [-4.448014584613572, -4.448033275602934, -4.448033545573118, -4.448033548267992, -4.448033548268098, -4.448033548268095, -4.448033548268104, -4.448033548268106],
})

# Numerical-refinement checks.
protocol_dt = pd.DataFrame({
    "tau_exp_Jinv": [5.0, 20.0, 80.0],
    "relative_paired_shift_percent": [-0.002537137531992953, -0.03243470133460129, 0.10860844644217629],
})
protocol_n = pd.DataFrame({
    "tau_exp_Jinv": [5.0, 20.0, 80.0],
    "relative_shift_percent": [0.0798438846891193, 0.013180512636572822, 0.3218066541735912],
})
l6_n = pd.DataFrame({
    "Q": [3, 4],
    "relative_shift_percent": [-0.4284430769871129, -0.36520032057754065],
})
l6_dt = pd.DataFrame({
    "Q": [3, 4],
    "relative_shift_percent": [-0.0212274215851451, -0.02066282428042579],
})

# =============================================================================
# Small data helpers
# =============================================================================


def mechanism_row(name: str) -> pd.Series:
    out = mechanism.loc[mechanism["case"] == name]
    if len(out) != 1:
        raise RuntimeError(f"Expected exactly one mechanism row for {name!r}.")
    return out.iloc[0]


def summary_value(quantity: str):
    out = matched_final.loc[matched_final["quantity"] == quantity]
    if len(out) != 1:
        raise RuntimeError(f"Expected exactly one summary row for {quantity!r}.")
    row = out.iloc[0]
    return float(row["value"]), float(row["standard_error"])


thermal_q = mechanism_row("thermal_quantum")
thermal_c = mechanism_row("thermal_classical")
matched_q = mechanism_row("matched_pE_quantum")
matched_c = mechanism_row("matched_pE_classical")

pi_q4 = exact_pi_L4_Q4()

# Endpoint p_C for the strict matched-(p_B,E_B) comparison.
# The quantum trajectory is the same Q=4 trajectory used in the thermal benchmark;
# the classical endpoint below comes from the final target-adapted matched-preparation run.
row_tau20 = protocol.loc[np.isclose(protocol["tau_exp_Jinv"], 20.0)].iloc[0]
pC_q = np.array([row_tau20[f"pC_quantum_{r}"] for r in range(5)], dtype=float)
matched_power16 = matched_conv.loc[matched_conv["power"] == 16].iloc[0]
pC_c = np.array([matched_power16[f"pC_{r}_mean"] for r in range(5)], dtype=float)
pC_c_se = np.array([matched_power16[f"pC_{r}_se"] for r in range(5)], dtype=float)

Dq_endpoint = relative_entropy_discrete(pC_q, pi_q4)
Dc_endpoint = relative_entropy_discrete(pC_c, pi_q4)

sigma_q_matched, _ = summary_value("sigma_BC_quantum")
sigma_c_matched, sigma_c_matched_se = summary_value("sigma_BC_classical_matched_preparation")
reduction_matched, _ = summary_value("quantum_reduction_percent_relative_to_classical")


# =============================================================================
# Retained-state and physical cycle bookkeeping
# =============================================================================

@dataclass(frozen=True)
class CycleLedgerAudit:
    """Two complementary ledgers for the constructive cycle.

    The retained ledger assigns Sigma_BC when the isolated dynamics creates a
    mismatch between the exact state and the (E,R) representative.  The physical
    ledger locates the corresponding microscopic entropy increase in the later
    C^+ -> Cbar relaxation.  They are the same irreversibility and must not be
    added together.
    """

    sigma_AB_retained: float
    sigma_BC_retained: float
    sigma_cold_isotherm_retained: float
    sigma_DA_retained: float
    sigma_cycle_retained: float
    sigma_BC_fine_physical: float
    sigma_switch_physical: float
    sigma_reset_physical: float
    sigma_cold_isotherm_physical: float
    sigma_DA_physical: float
    sigma_cycle_physical: float
    reservoir_entropy_change: float
    retained_balance_error: float
    physical_balance_error: float
    W_net_from_heat: float
    W_net_from_entropy: float
    work_identity_error: float
    eta: float


def constructive_cycle_ledger(
    *,
    S_A: float,
    S_B: float,
    S_C: float,
    S_D: float,
    Q_h: float,
    Q_c: float,
    T_h: float,
    T_c: float,
) -> CycleLedgerAudit:
    """Audit the retained and physical entropy ledgers of the revised cycle.

    S_C is S_{E,R}(C), i.e. the entropy of the energy-and-record
    representative.  The exact fine entropy at C^- still equals the fine
    entropy at B because B -> C^- is isolated and information preserving.
    """
    if T_h <= 0.0 or T_c <= 0.0:
        raise ValueError("Reservoir temperatures must be positive.")
    if Q_h <= 0.0:
        raise ValueError("Q_h must be positive for the heat-engine convention.")

    # Retained (E,R) thermodynamic ledger.
    sigma_AB = S_B - S_A - Q_h / T_h
    sigma_BC = S_C - S_B
    sigma_cold = S_D - S_C + Q_c / T_c
    sigma_DA = S_A - S_D
    sigma_retained = sigma_AB + sigma_BC + sigma_cold + sigma_DA

    # Physical microscopic realization.  The isolated stroke preserves fine
    # entropy.  The work-only switch preserves the state.  The fixed-H_{C,R}
    # reset increases the microscopic entropy by exactly the same Sigma_BC.
    sigma_BC_fine = 0.0
    sigma_switch = 0.0
    sigma_reset = sigma_BC
    sigma_cold_physical = 0.0
    sigma_DA_physical = 0.0
    sigma_physical = (
        sigma_BC_fine
        + sigma_switch
        + sigma_reset
        + sigma_cold_physical
        + sigma_DA_physical
    )

    reservoir_entropy = -Q_h / T_h + Q_c / T_c
    retained_error = sigma_retained - reservoir_entropy
    physical_error = sigma_physical - reservoir_entropy

    W_net_heat = Q_h - Q_c
    W_net_entropy = Q_h * (1.0 - T_c / T_h) - T_c * sigma_BC
    work_error = W_net_heat - W_net_entropy

    return CycleLedgerAudit(
        sigma_AB_retained=float(sigma_AB),
        sigma_BC_retained=float(sigma_BC),
        sigma_cold_isotherm_retained=float(sigma_cold),
        sigma_DA_retained=float(sigma_DA),
        sigma_cycle_retained=float(sigma_retained),
        sigma_BC_fine_physical=float(sigma_BC_fine),
        sigma_switch_physical=float(sigma_switch),
        sigma_reset_physical=float(sigma_reset),
        sigma_cold_isotherm_physical=float(sigma_cold_physical),
        sigma_DA_physical=float(sigma_DA_physical),
        sigma_cycle_physical=float(sigma_physical),
        reservoir_entropy_change=float(reservoir_entropy),
        retained_balance_error=float(retained_error),
        physical_balance_error=float(physical_error),
        W_net_from_heat=float(W_net_heat),
        W_net_from_entropy=float(W_net_entropy),
        work_identity_error=float(work_error),
        eta=float(W_net_heat / Q_h),
    )


# =============================================================================
# Exact constructive Q=4 quantum cycle audit
# =============================================================================

@dataclass(frozen=True)
class QuantumConstructiveAudit:
    sigma_BC: float
    S_fine_C: float
    S_ER_C: float
    S_R_C: float
    nested_decomposition_error: float
    chi_R_record_residual: float
    chi_ER_record_residual: float
    chi_ER_energy_residual: float
    ER_constraint_residual: float
    ER_is_Gibbs_residual: float
    HCR_energy_identity_error: float
    W_AB_on: float
    W_BC_on: float
    W_switch_on: float
    W_reset_on: float
    Q_reset_to_system: float
    W_CbarD_on: float
    W_DA_on: float
    Q_h: float
    Q_c: float
    W_net_direct: float
    W_net_from_heat: float
    work_closure_error: float
    reservoir_entropy_change: float
    entropy_closure_error: float
    eta: float
    beta_C_star: float
    alphas_C: tuple


def _fixed_Q_basis(L: int, Q: int):
    basis = []

    def rec(prefix, remaining, sites_left):
        if sites_left == 1:
            basis.append(tuple(prefix + [remaining]))
            return
        for n in range(remaining + 1):
            rec(prefix + [n], remaining - n, sites_left - 1)

    rec([], Q, L)
    return basis


def _build_quantum_Q4_model():
    L = 4
    Q = 4
    J = 1.0
    g = 1.3
    U = g / (Q - 1)
    basis = _fixed_Q_basis(L, Q)
    index = {state: i for i, state in enumerate(basis)}
    dim = len(basis)

    H_hop = np.zeros((dim, dim), dtype=complex)
    H_int = np.zeros((dim, dim), dtype=complex)
    N_R = np.zeros((dim, dim), dtype=complex)

    for i, state in enumerate(basis):
        N_R[i, i] = sum(state[L // 2 :])
        H_int[i, i] = 0.5 * U * sum(n * (n - 1) for n in state)
        for j in range(L - 1):
            if state[j + 1] > 0:
                target = list(state)
                amp = math.sqrt((state[j] + 1) * state[j + 1])
                target[j] += 1
                target[j + 1] -= 1
                H_hop[index[tuple(target)], i] += -J * amp
            if state[j] > 0:
                target = list(state)
                amp = math.sqrt((state[j + 1] + 1) * state[j])
                target[j] -= 1
                target[j + 1] += 1
                H_hop[index[tuple(target)], i] += -J * amp

    H0 = H_hop + H_int
    projectors = []
    for r in range(Q + 1):
        diag = [1.0 if sum(state[L // 2 :]) == r else 0.0 for state in basis]
        projectors.append(np.diag(diag).astype(complex))

    return H0, N_R, projectors


def _gibbs_density(H, T):
    vals, vecs = eigh(H)
    shifted = vals - np.min(vals)
    weights = np.exp(-shifted / T)
    weights /= np.sum(weights)
    return (vecs * weights) @ vecs.conj().T


def _von_neumann_entropy(rho):
    vals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real
    vals = vals[vals > 1.0e-15]
    return float(-np.sum(vals * np.log(vals)))


def _expectation(H, rho):
    return float(np.trace(H @ rho).real)


def _record_probabilities(rho, projectors):
    return np.array([np.trace(P @ rho).real for P in projectors], dtype=float)


def _thermodynamic_work_quantum(H_start, H_end, T, order=96):
    """Work done on the system along a quasistatic linear Hamiltonian path."""
    nodes, weights = leggauss(order)
    ss = 0.5 * (nodes + 1.0)
    ww = 0.5 * weights
    dH = H_end - H_start
    integrand = []
    for s in ss:
        Hs = (1.0 - s) * H_start + s * H_end
        rho_s = _gibbs_density(Hs, T)
        integrand.append(np.trace(rho_s @ dH).real)
    return float(np.dot(ww, np.asarray(integrand)))


def _fit_quantum_ER(H0, projectors, E_target, p_target):
    """Fit bar(rho)_{E,R}; use alpha_Q=0 as the fixed additive gauge."""
    nrec = len(projectors)

    def state_and_logZ(x):
        beta = float(x[0])
        alphas = np.concatenate([np.asarray(x[1:], dtype=float), [0.0]])
        K = beta * H0.copy()
        for r, P in enumerate(projectors):
            K = K + alphas[r] * P
        vals, vecs = eigh(K)
        m = float(np.min(vals))
        raw = np.exp(-(vals - m))
        Zshift = float(np.sum(raw))
        rho = (vecs * (raw / Zshift)) @ vecs.conj().T
        logZ = -m + math.log(Zshift)
        return rho, logZ, alphas

    def objective(x):
        rho, logZ, alphas = state_and_logZ(x)
        return float(logZ + x[0] * E_target + np.dot(alphas, p_target))

    def gradient(x):
        rho, _, _ = state_and_logZ(x)
        p = _record_probabilities(rho, projectors)
        return np.concatenate(
            [[E_target - _expectation(H0, rho)], p_target[:-1] - p[:-1]]
        )

    best = None
    for guess in (
        np.zeros(nrec),
        np.array([0.5] + [0.0] * (nrec - 1)),
        np.array([1.0] + [0.0] * (nrec - 1)),
    ):
        result = minimize(
            objective,
            guess,
            jac=gradient,
            method="BFGS",
            options={"gtol": 1.0e-11, "maxiter": 5000},
        )
        residual = float(np.max(np.abs(gradient(result.x))))
        if best is None or residual < best[0]:
            best = (residual, result)

    residual, result = best
    rho, _, alphas = state_and_logZ(result.x)
    return rho, float(result.x[0]), alphas, residual


def run_exact_quantum_cycle_audit() -> QuantumConstructiveAudit:
    """Rebuild and independently audit the representative L=4,Q=4 quantum cycle."""
    Q = 4
    Lambda0 = 20.0
    tau_BC = 20.0
    T_h = 5.0
    T_c = 1.5
    kappa_c = 2.7

    H0, N_R, projectors = _build_quantum_Q4_model()
    H_B = H0 + Lambda0 * N_R
    rho_B = _gibbs_density(H_B, T_h)
    S_B = _von_neumann_entropy(rho_B)
    E_B = _expectation(H_B, rho_B)

    vals, vecs = eigh(rho_B)
    vals = np.clip(vals.real, 0.0, None)
    A0 = (vecs * np.sqrt(vals)) @ vecs.conj().T
    dim = H0.shape[0]

    def rhs(t, y):
        A = y.reshape(dim, dim)
        Lambda = Lambda0 * (1.0 - t / tau_BC)
        H = H0 + Lambda * N_R
        return (-1j * H @ A).reshape(-1)

    sol = solve_ivp(
        rhs,
        (0.0, tau_BC),
        A0.reshape(-1),
        method="DOP853",
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    A_C = sol.y[:, -1].reshape(dim, dim)
    rho_C = A_C @ A_C.conj().T
    rho_C = 0.5 * (rho_C + rho_C.conj().T)
    rho_C /= np.trace(rho_C)

    E_C = _expectation(H0, rho_C)
    p_C = _record_probabilities(rho_C, projectors)
    S_fine_C = _von_neumann_entropy(rho_C)

    omegas = np.array([np.trace(P).real for P in projectors])
    rho_R = sum((p_C[r] / omegas[r]) * projectors[r] for r in range(Q + 1))
    S_R_C = _von_neumann_entropy(rho_R)

    rho_ER, beta_star, alphas, er_residual = _fit_quantum_ER(
        H0, projectors, E_C, p_C
    )
    S_ER_C = _von_neumann_entropy(rho_ER)
    sigma_BC = S_ER_C - S_B

    chi_R = rho_C - rho_R
    chi_ER = rho_C - rho_ER
    delta_E_given_R = rho_ER - rho_R
    nested_error = float(np.linalg.norm(chi_R - delta_E_given_R - chi_ER))
    chi_R_record = max(abs(np.trace(P @ chi_R)) for P in projectors)
    chi_ER_record = max(abs(np.trace(P @ chi_ER)) for P in projectors)
    chi_ER_energy = abs(np.trace(H0 @ chi_ER))

    H_CR = T_c * (
        beta_star * H0
        + sum(alphas[r] * projectors[r] for r in range(Q + 1))
    )
    rho_gibbs_CR = _gibbs_density(H_CR, T_c)
    gibbs_residual = float(np.linalg.norm(rho_ER - rho_gibbs_CR))
    HCR_energy_actual = _expectation(H_CR, rho_C)
    HCR_energy_rep = _expectation(H_CR, rho_ER)
    energy_identity_error = HCR_energy_actual - HCR_energy_rep
    W_switch = HCR_energy_actual - E_C

    H_D = kappa_c * H0
    H_A = (T_h / T_c) * H_D
    rho_D = _gibbs_density(H_D, T_c)
    rho_A = _gibbs_density(H_A, T_h)
    E_D = _expectation(H_D, rho_D)
    E_A = _expectation(H_A, rho_A)
    S_D = _von_neumann_entropy(rho_D)

    W_AB = _thermodynamic_work_quantum(H_A, H_B, T_h)
    Q_h = (E_B - E_A) - W_AB
    W_BC = E_C - E_B

    # C^+ -> Cbar: H fixed, endpoint energies equal, so W=0 and net Q=0.
    W_reset = 0.0
    Q_reset = HCR_energy_rep - HCR_energy_actual

    W_CbarD = _thermodynamic_work_quantum(H_CR, H_D, T_c)
    Q_to_system_cold = (E_D - HCR_energy_rep) - W_CbarD
    Q_c = -Q_to_system_cold

    # State is unchanged on D -> A, so work on the system is the energy change.
    W_DA = E_A - E_D

    W_on_total = W_AB + W_BC + W_switch + W_reset + W_CbarD + W_DA
    W_net_direct = -W_on_total
    W_net_heat = Q_h - Q_c
    work_error = W_net_direct - W_net_heat
    reservoir_entropy = -Q_h / T_h + Q_c / T_c
    entropy_error = reservoir_entropy - sigma_BC

    # Guard the identities that the revised manuscript relies on.
    if nested_error > 1.0e-10:
        raise RuntimeError("Quantum nested-state decomposition failed.")
    if max(chi_R_record, chi_ER_record, chi_ER_energy) > 5.0e-8:
        raise RuntimeError("Quantum unresolved-component constraints failed.")
    if er_residual > 5.0e-8:
        raise RuntimeError("Quantum (E,R) reconstruction failed.")
    if gibbs_residual > 1.0e-10:
        raise RuntimeError("bar(rho)_{E,R}(C) is not Gibbs for H_{C,R} at T_c.")
    if abs(energy_identity_error) > 5.0e-8:
        raise RuntimeError("Quantum H_{C,R} endpoint-energy identity failed.")
    if abs(Q_reset) > 5.0e-8:
        raise RuntimeError("Quantum C^+ -> Cbar net-heat identity failed.")
    if abs((S_ER_C - S_fine_C) - sigma_BC) > 5.0e-8:
        raise RuntimeError("Quantum reset entropy does not equal Sigma_BC.")
    if abs(work_error) > 5.0e-8:
        raise RuntimeError("Quantum direct-work cycle closure failed.")
    if abs(entropy_error) > 5.0e-8:
        raise RuntimeError("Quantum cycle entropy closure failed.")
    if abs(S_D - _von_neumann_entropy(rho_A)) > 1.0e-10:
        raise RuntimeError("Quantum D -> A equilibrium-state scaling failed.")

    return QuantumConstructiveAudit(
        sigma_BC=float(sigma_BC),
        S_fine_C=float(S_fine_C),
        S_ER_C=float(S_ER_C),
        S_R_C=float(S_R_C),
        nested_decomposition_error=float(nested_error),
        chi_R_record_residual=float(chi_R_record),
        chi_ER_record_residual=float(chi_ER_record),
        chi_ER_energy_residual=float(chi_ER_energy),
        ER_constraint_residual=float(er_residual),
        ER_is_Gibbs_residual=float(gibbs_residual),
        HCR_energy_identity_error=float(energy_identity_error),
        W_AB_on=float(W_AB),
        W_BC_on=float(W_BC),
        W_switch_on=float(W_switch),
        W_reset_on=float(W_reset),
        Q_reset_to_system=float(Q_reset),
        W_CbarD_on=float(W_CbarD),
        W_DA_on=float(W_DA),
        Q_h=float(Q_h),
        Q_c=float(Q_c),
        W_net_direct=float(W_net_direct),
        W_net_from_heat=float(W_net_heat),
        work_closure_error=float(work_error),
        reservoir_entropy_change=float(reservoir_entropy),
        entropy_closure_error=float(entropy_error),
        eta=float(W_net_direct / Q_h),
        beta_C_star=float(beta_star),
        alphas_C=tuple(float(a) for a in alphas),
    )


# =============================================================================
# Optional full classical microscopic closure audit
# =============================================================================

@dataclass(frozen=True)
class ClassicalConstructiveAudit:
    sigma_BC: float
    beta_C_star: float
    alphas_C: tuple
    endpoint_energy_per_particle: float
    endpoint_probabilities: tuple
    endpoint_energy_se: float
    endpoint_probability_se_max: float
    ER_energy_residual: float
    HCR_energy_identity_error: float
    Q_reset_to_system: float
    W_switch_on: float
    W_AB_on: float
    W_BC_on: float
    W_CbarD_on: float
    W_DA_on: float
    Q_h: float
    Q_c: float
    W_net_direct: float
    W_net_from_heat: float
    work_closure_error: float
    reservoir_entropy_change: float
    entropy_closure_error: float
    eta: float


def _classical_record_boundaries_Q4():
    return np.array([
        0.0,
        0.237896765756007,
        0.413420377776432,
        0.586579622223568,
        0.762103234243993,
        1.0,
    ])


def _classical_h0(z, g=1.3):
    hop = -2.0 * np.sum(np.real(np.conj(z[:, :-1]) * z[:, 1:]), axis=1)
    interaction = 0.5 * g * np.sum(np.abs(z) ** 4, axis=1)
    return hop + interaction


def _classical_xR(z):
    return np.sum(np.abs(z[:, 2:]) ** 2, axis=1)


def _classical_bin_index(z, boundaries):
    return np.searchsorted(boundaries[1:-1], _classical_xR(z), side="right")


def _sobol_CP3(power, seed):
    sampler = qmc.Sobol(d=8, scramble=True, seed=seed)
    u = sampler.random_base2(power)
    u = np.clip(u, 1.0e-12, 1.0 - 1.0e-12)
    xi = ndtri(u)
    z = xi[:, 0::2] + 1j * xi[:, 1::2]
    z /= np.linalg.norm(z, axis=1)[:, None]
    return z


def _propagate_classical_DNLS(z, tau_BC=20.0, dt=0.05, Lambda0=20.0, g=1.3):
    z = np.asarray(z, dtype=complex).copy()
    L = z.shape[1]
    H_hop = np.zeros((L, L), dtype=complex)
    for j in range(L - 1):
        H_hop[j, j + 1] = H_hop[j + 1, j] = -1.0
    U_hop = expm(-1j * H_hop * dt)
    chi = np.array([0.0, 0.0, 1.0, 1.0])
    n_steps = int(round(tau_BC / dt))
    for n in range(n_steps):
        t_mid = (n + 0.5) * dt
        Lambda = Lambda0 * (1.0 - t_mid / tau_BC)
        z *= np.exp(-1j * (g * np.abs(z) ** 2 + Lambda * chi) * dt / 2.0)
        z = z @ U_hop.T
        z *= np.exp(-1j * (g * np.abs(z) ** 2 + Lambda * chi) * dt / 2.0)
    return z


def _build_classical_quadrature_cache(order=16, g=1.3):
    """Precompute the phase-reduced CP^3 quadrature used by the classical audit."""
    boundaries = _classical_record_boundaries_Q4()
    nodes, weights = leggauss(order)
    u = 0.5 * (nodes + 1.0)
    wu = 0.5 * weights
    U, V = np.meshgrid(u, u, indexing="ij")
    U = U.ravel()
    V = V.ravel()
    WUV = np.outer(wu, wu).ravel()

    xs, vints, roots_all, base_all, region_all = [], [], [], [], []
    for r in range(5):
        a, b = boundaries[r], boundaries[r + 1]
        x_nodes = 0.5 * (b - a) * (nodes + 1.0) + a
        x_weights = 0.5 * (b - a) * weights
        for x, wx in zip(x_nodes, x_weights):
            p = np.stack(
                [
                    (1.0 - x) * U,
                    (1.0 - x) * (1.0 - U),
                    x * V,
                    x * (1.0 - V),
                ],
                axis=1,
            )
            vint = 0.5 * g * np.sum(p * p, axis=1)
            roots = np.sqrt(p[:, :-1] * p[:, 1:])
            base = 6.0 * x * (1.0 - x) * wx * WUV
            xs.append(np.full(vint.shape, x))
            vints.append(vint)
            roots_all.append(roots)
            base_all.append(base)
            region_all.append(np.full(vint.shape, r, dtype=int))

    return {
        "x": np.concatenate(xs),
        "vint": np.concatenate(vints),
        "roots": np.concatenate(roots_all),
        "base": np.concatenate(base_all),
        "region": np.concatenate(region_all),
    }


def _classical_bin_integrals(cache, beta_h0, lambda_x=0.0):
    x = cache["x"]
    vint = cache["vint"]
    roots = cache["roots"]
    base = cache["base"]
    region = cache["region"]

    kappa = 2.0 * beta_h0 * roots
    I0 = i0(kappa)
    phase_factor = np.prod(I0, axis=1)
    weight = base * np.exp(-beta_h0 * vint - lambda_x * x) * phase_factor
    phase_cos = i1(kappa) / I0
    h0_phase_avg = vint - 2.0 * np.sum(roots * phase_cos, axis=1)

    Z = np.bincount(region, weights=weight, minlength=5)
    H = np.bincount(region, weights=weight * h0_phase_avg, minlength=5)
    X = np.bincount(region, weights=weight * x, minlength=5)
    return Z, H, X


def run_full_classical_cycle_audit(
    *,
    qmc_power=16,
    scrambles=8,
    dt=0.05,
    quadrature_order=16,
) -> ClassicalConstructiveAudit:
    """Recompute a natural Q=4 classical endpoint and construct its cycle closure.

    This is intentionally optional because the DNLS propagation is the expensive
    part of the original calculation.  It is an independent rerun and therefore
    fluctuates within QMC uncertainty around the embedded publication means.
    """
    Q = 4
    theta_h = 1.25
    theta_c = 0.375
    T_h = Q * theta_h
    T_c = Q * theta_c
    Lambda0 = 20.0
    tau_BC = 20.0
    kappa_c = 2.564983
    boundaries = _classical_record_boundaries_Q4()
    cache = _build_classical_quadrature_cache(quadrature_order)

    def full_stats(beta_h0, lambda_x=0.0):
        Zr, Hr, Xr = _classical_bin_integrals(cache, beta_h0, lambda_x)
        Z = float(np.sum(Zr))
        return Z, float(np.sum(Hr) / Z), float(np.sum(Xr) / Z), Zr, Hr, Xr

    # Deterministic equilibrium thermodynamics for the chosen Hamiltonian paths.
    Z_B, h0_B, x_B, _, _, _ = full_stats(1.0 / theta_h, Lambda0 / theta_h)
    h_B_mean = h0_B + Lambda0 * x_B
    S_B = math.log(Z_B) + h_B_mean / theta_h

    Z_D, h0_D, _, _, _, _ = full_stats(kappa_c / theta_c, 0.0)
    h_D_mean = kappa_c * h0_D
    S_D = math.log(Z_D) + h_D_mean / theta_c
    a_A = (theta_h / theta_c) * kappa_c

    # Explicit A -> B thermodynamic integration.
    nodes, weights = leggauss(48)
    ss = 0.5 * (nodes + 1.0)
    ww = 0.5 * weights
    W_AB = 0.0
    for s, ws in zip(ss, ww):
        a = (1.0 - s) * a_A + s
        Lambda = s * Lambda0
        _, mean_h0, mean_x, _, _, _ = full_stats(a / theta_h, Lambda / theta_h)
        W_AB += ws * Q * ((1.0 - a_A) * mean_h0 + Lambda0 * mean_x)

    E_A = Q * a_A * h0_D
    E_B = Q * h_B_mean
    Q_h = (E_B - E_A) - W_AB

    # Actual microscopic endpoint C^- from the natural canonical ensemble.
    rows = []
    for k in range(scrambles):
        z = _sobol_CP3(qmc_power, seed=20260908 + 1009 * k)
        h_B_samples = _classical_h0(z) + Lambda0 * _classical_xR(z)
        logw = -h_B_samples / theta_h
        logw -= np.max(logw)
        w = np.exp(logw)
        w /= np.sum(w)
        z_C = _propagate_classical_DNLS(z, tau_BC=tau_BC, dt=dt, Lambda0=Lambda0)
        e_C = float(np.dot(w, _classical_h0(z_C)))
        bins_C = _classical_bin_index(z_C, boundaries)
        p_C = np.array([np.sum(w[bins_C == r]) for r in range(5)], dtype=float)
        rows.append(np.concatenate([[e_C], p_C]))

    rows = np.asarray(rows)
    mean = np.mean(rows, axis=0)
    if scrambles > 1:
        se = np.std(rows, axis=0, ddof=1) / math.sqrt(scrambles)
    else:
        se = np.zeros_like(mean)
    e_C = float(mean[0])
    p_C = np.asarray(mean[1:], dtype=float)

    # Energy-and-record maximum-entropy representative at C.
    def energy_residual(beta):
        Zr, Hr, _ = _classical_bin_integrals(cache, beta, 0.0)
        return float(np.dot(p_C, Hr / Zr) - e_C)

    grid = np.linspace(-20.0, 20.0, 161)
    vals = np.array([energy_residual(b) for b in grid])
    bracket = None
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if fa == 0.0 or fa * fb < 0.0:
            bracket = (a, b)
            break
    if bracket is None:
        raise RuntimeError("Could not bracket the classical (E,R) beta multiplier.")
    beta_star = brentq(energy_residual, *bracket, xtol=1.0e-13)
    Zr, Hr, _ = _classical_bin_integrals(cache, beta_star, 0.0)
    e_r = Hr / Zr
    er_residual = float(np.dot(p_C, e_r) - e_C)
    S_ER_C = float(
        -np.sum(p_C * np.log(p_C))
        + np.sum(p_C * np.log(Zr))
        + beta_star * e_C
    )
    sigma_BC = S_ER_C - S_B

    # Fix the same alpha_Q=0 gauge used in the quantum audit.
    alphas = np.log(Zr / p_C) - math.log(Zr[-1] / p_C[-1])
    hCR_mean_actual = theta_c * (beta_star * e_C + np.dot(alphas, p_C))
    hCR_mean_rep = theta_c * (beta_star * np.dot(p_C, e_r) + np.dot(alphas, p_C))
    hcr_identity_error = hCR_mean_actual - hCR_mean_rep
    W_switch = Q * (hCR_mean_actual - e_C)
    Q_reset = Q * (hCR_mean_rep - hCR_mean_actual)

    # Genuine reversible Cbar -> D isotherm.
    W_CbarD = 0.0
    for s, ws in zip(ss, ww):
        beta_eff = (1.0 - s) * beta_star + s * kappa_c / theta_c
        Zs, Hs, _ = _classical_bin_integrals(cache, beta_eff, 0.0)
        sector_factor = np.exp(-(1.0 - s) * alphas)
        Ztot = float(np.dot(sector_factor, Zs))
        mean_h0 = float(np.dot(sector_factor, Hs) / Ztot)
        mean_alpha = float(np.dot(sector_factor * Zs, alphas) / Ztot)
        dhds = (kappa_c - theta_c * beta_star) * mean_h0 - theta_c * mean_alpha
        W_CbarD += ws * Q * dhds

    E_D = Q * h_D_mean
    Q_to_system_cold = (E_D - Q * hCR_mean_rep) - W_CbarD
    Q_c = -Q_to_system_cold

    W_BC = Q * (e_C - h_B_mean)
    W_DA = E_A - E_D
    W_on_total = W_AB + W_BC + W_switch + W_CbarD + W_DA
    W_net_direct = -W_on_total
    W_net_heat = Q_h - Q_c
    work_error = W_net_direct - W_net_heat
    reservoir_entropy = -Q_h / T_h + Q_c / T_c
    entropy_error = reservoir_entropy - sigma_BC

    # These are structural closure tests.  Agreement with the embedded published
    # ensemble means is assessed separately because a fresh QMC rerun fluctuates.
    if abs(er_residual) > 2.0e-7:
        raise RuntimeError("Classical (E,R) energy fit failed.")
    if abs(hcr_identity_error) > 2.0e-7:
        raise RuntimeError("Classical H_{C,R} endpoint-energy identity failed.")
    if abs(Q_reset) > 1.0e-6:
        raise RuntimeError("Classical C^+ -> Cbar net-heat identity failed.")
    if abs(work_error) > 2.0e-5:
        raise RuntimeError("Classical direct-work cycle closure failed.")
    if abs(entropy_error) > 2.0e-5:
        raise RuntimeError("Classical cycle entropy closure failed.")

    return ClassicalConstructiveAudit(
        sigma_BC=float(sigma_BC),
        beta_C_star=float(beta_star),
        alphas_C=tuple(float(a) for a in alphas),
        endpoint_energy_per_particle=float(e_C),
        endpoint_probabilities=tuple(float(p) for p in p_C),
        endpoint_energy_se=float(se[0]),
        endpoint_probability_se_max=float(np.max(se[1:])),
        ER_energy_residual=float(er_residual),
        HCR_energy_identity_error=float(hcr_identity_error),
        Q_reset_to_system=float(Q_reset),
        W_switch_on=float(W_switch),
        W_AB_on=float(W_AB),
        W_BC_on=float(W_BC),
        W_CbarD_on=float(W_CbarD),
        W_DA_on=float(W_DA),
        Q_h=float(Q_h),
        Q_c=float(Q_c),
        W_net_direct=float(W_net_direct),
        W_net_from_heat=float(W_net_heat),
        work_closure_error=float(work_error),
        reservoir_entropy_change=float(reservoir_entropy),
        entropy_closure_error=float(entropy_error),
        eta=float(W_net_direct / Q_h),
    )


# =============================================================================
# Figure 1: matched quantum/classical framework and information geometry
# =============================================================================


def make_figure_1():
    fig = plt.figure(figsize=(13.5, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.1], wspace=0.2)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # The default GridSpec spacing makes the visual gap between panels (b) and (c)
    # appear larger than that between (a) and (b), mainly because panel (b) carries
    # axis decorations while panel (c) is a schematic with the axis turned off.
    # To equalize the visual spacing without touching anything else, shift panel (c)
    # slightly left while keeping its size unchanged.
    pos_c = axes[2].get_position()
    axes[2].set_position([pos_c.x0 - 0.028, pos_c.y0, pos_c.width, pos_c.height])

    # ------------------------------------------------------------------
    # (a) Matched Bose-Hubbard / DNLS dynamics
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.set_axis_off()

    qbox = FancyBboxPatch(
        (0.06, 0.62), 0.89, 0.24,
        boxstyle="round,pad=0.025,rounding_size=0.03",
        facecolor="white", edgecolor=RED, linewidth=1.5,
        transform=ax.transAxes,
    )
    cbox = FancyBboxPatch(
        (0.06, 0.19), 0.89, 0.24,
        boxstyle="round,pad=0.025,rounding_size=0.03",
        facecolor="white", edgecolor=NAVY, linewidth=1.5,
        transform=ax.transAxes,
    )
    ax.add_patch(qbox)
    ax.add_patch(cbox)

    ax.text(0.50, 0.78, "Bose--Hubbard", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.50, 0.685, r"$U=g/(Q-1)$,  $\Lambda(t)$",
        ha="center", va="center", transform=ax.transAxes,
    )
    ax.text(0.50, 0.35, "Classical DNLS", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.50, 0.255, r"$i\dot z_j=-J(z_{j-1}+z_{j+1})+g|z_j|^2z_j+\Lambda\chi_R z_j$",
        ha="center", va="center", transform=ax.transAxes, fontsize=12,
    )

    arrow = FancyArrowPatch(
        (0.50, 0.60), (0.50, 0.45),
        arrowstyle="<->", mutation_scale=11, linewidth=1.2, color=BLACK,
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)
    ax.text(0.44, 0.525, r"$Q\uparrow$", ha="right", va="center", transform=ax.transAxes, fontsize=12)
    ax.text(0.50, 0.03, r"same record $R=n_R$" + "\n" + r"same confinement/work protocol $\Lambda(t)$",
        ha="center", va="center", transform=ax.transAxes, fontsize=12, linespacing=1.25,
    )
    panel_label(ax, "(a)", x=0.02, y=1)

    # ------------------------------------------------------------------
    # (b) Matched reference measure pi
    # ------------------------------------------------------------------
    ax = axes[1]
    x = np.linspace(0.0, 1.0, 1000)
    # Exact Beta(2,2) density for the uniform CP^3 right-half fraction.
    pdf = 6.0 * x * (1.0 - x)
    # Beta(2,2) quantiles of the cumulative Q=4 quantum multiplicities.
    # These boundaries enforce pi_R^cl = pi_R^Q in the continuum.
    boundaries = np.array([
        0.0,
        0.237896765756007,
        0.413420377776432,
        0.586579622223568,
        0.762103234243993,
        1.0,
    ])

    region_colors = [RED, CYAN, GREEN, NAVY, SALMON]
    ax.plot(x, pdf, color=BLACK, linewidth=1.5)
    for r in range(5):
        mask = (x >= boundaries[r]) & (x <= boundaries[r + 1])
        ax.fill_between(x[mask], 0.0, pdf[mask], color=region_colors[r], alpha=0.22, linewidth=0)
        if r > 0:
            ax.axvline(boundaries[r], color=GRAY, linestyle=":", linewidth=1.0)

    ax.set_xlabel(r"$x_R$")
    ax.set_ylabel(r"Liouville density")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.text(
        0.50, 0.85,
        r"$\pi_R^{\rm cl}=\pi_R^{\rm Q}$",
        transform=ax.transAxes, ha="center", va="top",
    )
    finish_axes(ax)
    panel_label(ax, "(b)")

    # ------------------------------------------------------------------
    # (c) Nested accessible-information geometry
    # ------------------------------------------------------------------
    ax = axes[2]
    ax.set_axis_off()

    outer = Ellipse(
        (0.50, 0.52), 0.95, 0.8,
        facecolor=LAVENDER, alpha=0.14, edgecolor=NAVY, linewidth=1.3,
        transform=ax.transAxes,
    )
    inner = Ellipse(
        (0.58, 0.52), 0.65, 0.5,
        facecolor=MINT, alpha=0.22, edgecolor=GREEN, linewidth=1.3,
        transform=ax.transAxes,
    )
    ax.add_patch(outer)
    ax.add_patch(inner)

    ax.text(0.50, 0.97, r"states compatible with $R$", ha="center", va="center", transform=ax.transAxes, fontsize=MAIN_TEXT_FONT_SIZE)
    ax.text(0.58, 0.8, r"compatible with $(E,R)$", ha="center", va="center", transform=ax.transAxes, fontsize=MAIN_TEXT_FONT_SIZE)

    # Actual microscopic state and the two maximum-entropy representatives.
    ax.plot(0.22, 0.49, marker="o", ms=5.2, color=NAVY, transform=ax.transAxes, clip_on=False)
    ax.plot(0.49, 0.54, marker="o", ms=5.2, color=GREEN, transform=ax.transAxes, clip_on=False)
    ax.plot(0.73, 0.45, marker="o", ms=5.2, color=RED, transform=ax.transAxes, clip_on=False)
    ax.text(0.18, 0.39, r"$\bar\rho_R$", ha="center", va="top", color=NAVY, transform=ax.transAxes, fontsize=MAIN_TEXT_FONT_SIZE)
    ax.text(0.49, 0.62, r"$\bar\rho_{E,R}$", ha="center", va="bottom", color=GREEN, transform=ax.transAxes, fontsize=MAIN_TEXT_FONT_SIZE)
    ax.text(0.77, 0.43, r"$\rho$", ha="center", va="top", color=RED, transform=ax.transAxes, fontsize=MAIN_TEXT_FONT_SIZE)

    ax.text(0.50, 0.05, r"$G_R=\mathcal{A}_E+\mathcal{I}_{E,R}$", ha="center", va="center", transform=ax.transAxes)
    #ax.text(0.50, 0.045, r"hidden by $R$ = energy-resolved + still inaccessible", ha="center", va="center", transform=ax.transAxes, fontsize=MAIN_TEXT_FONT_SIZE)
    panel_label(ax, "(c)", x=0.02, y=0.98)

    return savefig(fig, "Fig1_framework_and_geometry.pdf", w_pad=0.2)


# =============================================================================
# Figure 2: energy-resolved mechanism
# =============================================================================


def make_figure_2():
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2))

    # ------------------------------------------------------------------
    # (a) Strict matched-(p_B,E_B) comparison: Delta G_R, Delta A_E, Sigma_BC
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    cats = [r"$\Delta G_R$", r"$\Delta\mathcal{A}_E$", r"$\Sigma_{BC}$"]
    qvals = np.array([matched_q["DeltaG_R"], matched_q["DeltaI_E"], matched_q["Sigma_BC"]])
    cvals = np.array([matched_c["DeltaG_R"], matched_c["DeltaI_E"], matched_c["Sigma_BC"]])
    cerr = np.array([matched_c["DeltaG_R_se"], matched_c["DeltaI_E_se"], matched_c["Sigma_BC_se"]])

    xx = np.arange(len(cats), dtype=float)
    width = 0.34
    ax.bar(xx - width / 2, qvals, width=width, color=RED, label="Quantum")
    ax.bar(
        xx + width / 2, cvals, width=width, color=NAVY, label="Classical",
        yerr=cerr, capsize=2.5, error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax.set_xticks(xx, cats)
    ax.set_ylabel(r"Entropy change$/k_{\rm B}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1.0))
    finish_axes(ax, minor=False)
    panel_label(ax, "(a)", 0.03, 0.97)

    # ------------------------------------------------------------------
    # (b) Exact decomposition of the quantum-classical gap
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    record_term = float(matched_c["DeltaG_R"] - matched_q["DeltaG_R"])
    energy_term = float(-(matched_c["DeltaI_E"] - matched_q["DeltaI_E"]))
    total_gap = float(matched_c["Sigma_BC"] - matched_q["Sigma_BC"])
    record_se = float(matched_c["DeltaG_R_se"])
    energy_se = float(matched_c["DeltaI_E_se"])
    total_se = float(matched_c["Sigma_BC_se"])

    vals = [record_term, energy_term, total_gap]
    errs = [record_se, energy_se, total_se]
    labels = [r"Record $\Delta G_R$", r"Energy $\Delta\mathcal{A}_E$", r"Total $\Delta\Sigma_{BC}$"]
    colors = [LAVENDER, GREEN, NAVY]
    ax.bar(np.arange(3), vals, yerr=errs, capsize=2.5, width=0.64, color=colors)
    ax.axhline(0.0, linestyle=":", linewidth=1.2, color=GRAY)
    ax.set_xticks(np.arange(3), labels, rotation=12, ha="right")
    ax.set_ylabel(r"Classical $-$ quantum$/k_{\rm B}$")
    finish_axes(ax, minor=False)
    panel_label(ax, "(b)")

    # ------------------------------------------------------------------
    # (c) Endpoint macrodistribution and common pi
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    r = np.arange(5)
    ax.errorbar(
        r - 0.06, pC_c, yerr=pC_c_se, fmt="s", ms=4.3,
        color=NAVY, capsize=2.2, label="Classical", zorder=3,
    )
    ax.plot(r + 0.06, pC_q, "o", ms=4.6, color=RED, label="Quantum", zorder=4)
    ax.plot(r, pi_q4, "d--", ms=4.0, color=BLACK, linewidth=1.2, label=r"$\pi$")
    ax.set_xlabel(r"right-half occupation $R=n_R$")
    ax.set_ylabel(r"Probability")
    ax.set_xticks(r)
    # Horizontal legend placed directly on top of the panel-(c) plot box.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.005),
        ncol=3,
        fontsize=13,
        columnspacing=0.8,
        handlelength=1.8,
        borderaxespad=0.0,
    )
    finish_axes(ax)
    panel_label(ax, "(c)")

    # ------------------------------------------------------------------
    # (d) Preparation robustness: thermal vs strict matched-(p_B,E_B)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    labels = ["Thermal", r"Matched $(p_B,E_B)$"]
    qv = [thermal_q["Sigma_BC"], matched_q["Sigma_BC"]]
    cv = [thermal_c["Sigma_BC"], matched_c["Sigma_BC"]]
    ce = [thermal_c["Sigma_BC_se"], matched_c["Sigma_BC_se"]]
    xx = np.arange(2, dtype=float)
    width = 0.34
    ax.bar(xx - width / 2, qv, width=width, color=RED, label="Quantum")
    ax.bar(xx + width / 2, cv, width=width, color=NAVY, yerr=ce, capsize=2.5, label="Classical")
    ax.set_xticks(xx, labels, rotation=8)
    ax.set_ylabel(r"$\Sigma_{BC}/k_{\rm B}$")
    ax.legend(loc="upper center")
    finish_axes(ax, minor=False)
    panel_label(ax, "(d)")

    return savefig(fig, "Fig2_energy_resolved_mechanism.pdf", h_pad=0.8, w_pad=0.7)


# =============================================================================
# Figure 3: particle-number crossover and work consequence
# =============================================================================


def make_figure_3():
    """Complete-cycle particle-number crossover.

    For the constructive closure, the hot isotherm, Cbar -> D isotherm, and
    D -> A scaling are reversible.  The C^+ -> Cbar interface has zero net heat
    and no change of the retained (E,R) state; microscopically it realizes the
    same Sigma_BC already assigned to the isolated stroke.  Hence
    Sigma_cyc=Sigma_BC without double counting.
    """
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2))

    Q = qscale["Q"].to_numpy(dtype=float)
    scyc_q = qscale["sigma_cyc_q"].to_numpy(dtype=float)
    scyc_cl = qscale["sigma_cyc_cl"].to_numpy(dtype=float)
    scyc_se = qscale["sigma_cyc_gap_se"].to_numpy(dtype=float)
    gap_cyc = qscale["sigma_cyc_gap"].to_numpy(dtype=float)
    reduction = qscale["cycle_reduction_pct"].to_numpy(dtype=float)
    reduction_se = qscale["cycle_reduction_pct_se"].to_numpy(dtype=float)
    eta_adv = qscale["eta_adv_pp"].to_numpy(dtype=float)
    eta_adv_se = qscale["eta_adv_pp_se"].to_numpy(dtype=float)

    # (a) Complete-cycle entropy generation.
    ax = axes[0, 0]
    ax.plot(Q, scyc_q, "o-", color=RED, label="Quantum")
    ax.errorbar(Q, scyc_cl, yerr=scyc_se, fmt="s-", color=NAVY, capsize=2.2, label="Classical")
    ax.set_xlabel(r"Particle number $Q$")
    ax.set_ylabel(r"$\Sigma_{\rm cyc}/k_{\rm B}$")
    ax.set_xticks(Q.astype(int))
    ax.legend()
    finish_axes(ax)
    panel_label(ax, "(a)", 0.03, 0.90)

    # (b) Complete-cycle classical-minus-quantum entropy-generation gap.
    ax = axes[0, 1]
    ax.errorbar(Q, gap_cyc, yerr=scyc_se, fmt="o-", color=GREEN, capsize=2.2)
    ax.axhline(0.0, linestyle=":", linewidth=1.2, color=GRAY)
    ax.set_xlabel(r"Particle number $Q$")
    ax.set_ylabel(r"$(\Sigma_{\rm cyc}^{\rm cl}-\Sigma_{\rm cyc}^{\rm Q})/k_{\rm B}$")
    ax.set_xticks(Q.astype(int))
    finish_axes(ax)
    panel_label(ax, "(b)", 0.09, 0.96)

    # (c) Relative complete-cycle quantum reduction.
    ax = axes[1, 0]
    ax.errorbar(Q, reduction, yerr=reduction_se, fmt="d-", color=CYAN, capsize=2.2)
    ax.axhline(0.0, linestyle=":", linewidth=1.2, color=GRAY)
    ax.set_xlabel(r"Particle number $Q$")
    ax.set_ylabel(r"Cycle reduction $(\%)$")
    ax.set_xticks(Q.astype(int))
    finish_axes(ax)
    panel_label(ax, "(c)", 0.09, 0.96)

    # (d) Same-Q_h complete-cycle efficiency advantage.
    ax = axes[1, 1]
    ax.errorbar(Q, eta_adv, yerr=eta_adv_se, fmt="^-", color=NAVY, capsize=2.2)
    ax.axhline(0.0, linestyle=":", linewidth=1.2, color=GRAY)
    ax.set_xlabel(r"Particle number $Q$")
    ax.set_ylabel(r"$\eta_{\rm Q}-\eta_{\rm cl}$ (percentage points)")
    ax.set_xticks(Q.astype(int))
    finish_axes(ax)
    panel_label(ax, "(d)", 0.09, 0.96)

    return savefig(fig, "Fig3_quantum_classical_crossover.pdf", h_pad=0.8, w_pad=0.8)


# =============================================================================
# Figure 4: final robustness checks
# =============================================================================


def make_figure_4():
    """Complete-cycle protocol-duration robustness.

    Every value of tau_BC produces a different actual C^-, record-only
    representative, energy-and-record representative, and H_{C,R}.  The
    endpoint-specific physical continuation is C^- -> C^+ -> Cbar -> D.
    Panel (d) audits the independent (E,R) fits used to build H_{C,R}; it is not
    a simulation of the subsequent cold relaxation.

    A, B, and D are fixed throughout this scan, and D -> A is the same reversible
    isolated rescaling for every row.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.7))

    tau = protocol["tau_exp_Jinv"].to_numpy(dtype=float)
    scyc_q = protocol["sigma_cyc_q"].to_numpy(dtype=float)
    scyc_cl = protocol["sigma_cyc_cl"].to_numpy(dtype=float)
    scyc_se = protocol["sigma_cyc_gap_se"].to_numpy(dtype=float)
    gap_cyc = protocol["sigma_cyc_gap"].to_numpy(dtype=float)
    eta_pp = 100.0 * protocol["same_Qh_delta_eta_identity"].to_numpy(dtype=float)
    eta_pp_se = 100.0 * protocol["same_Qh_delta_eta_identity_se"].to_numpy(dtype=float)
    q_er = protocol["quantum_ER_residual"].to_numpy(dtype=float)
    c_er = protocol["classical_ER_max_residual"].to_numpy(dtype=float)

    # (a) Complete-cycle entropy generation.
    ax = axes[0, 0]
    ax.plot(tau, scyc_q, "o-", color=RED, label="Quantum")
    ax.errorbar(tau, scyc_cl, yerr=scyc_se, fmt="s-", color=NAVY, capsize=2.2, label="Classical")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Expansion duration $J\tau_{BC}$")
    ax.set_ylabel(r"$\Sigma_{\rm cyc}/k_{\rm B}$")
    ax.set_xticks(tau, [str(int(x)) for x in tau])
    ax.legend(loc="lower left")
    finish_axes(ax)
    panel_label(ax, "(a)", 0.88, 0.94)

    # (b) Complete-cycle efficiency advantage.
    ax = axes[0, 1]
    ax.errorbar(tau, eta_pp, yerr=eta_pp_se, fmt="o-", color=GREEN, capsize=2.2)
    ax.axhline(0.0, linestyle=":", linewidth=1.2, color=GRAY)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Expansion duration $J\tau_{BC}$")
    ax.set_ylabel(r"$\eta_{\rm Q}-\eta_{\rm cl}$ (percentage points)")
    ax.set_xticks(tau, [str(int(x)) for x in tau])
    finish_axes(ax)
    panel_label(ax, "(b)", 0.88, 0.94)

    # (c) Complete-cycle entropy-generation advantage.
    ax = axes[1, 0]
    ax.errorbar(tau, gap_cyc, yerr=scyc_se, fmt="o-", color=NAVY, capsize=2.2)
    ax.axhline(0.0, linestyle=":", linewidth=1.2, color=GRAY)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Expansion duration $J\tau_{BC}$")
    ax.set_ylabel(r"$(\Sigma_{\rm cyc}^{\rm cl}-\Sigma_{\rm cyc}^{\rm Q})/k_{\rm B}$")
    ax.set_xticks(tau, [str(int(x)) for x in tau])
    finish_axes(ax)
    panel_label(ax, "(c)", 0.88, 0.94)

    # (d) Numerical audit: each changed C is reconstructed independently.
    # A different fitted representative means a different H_{C,R}(tau_BC).
    # This panel checks only the endpoint reconstruction used to define the
    # constructive C^- -> C^+ -> Cbar -> D continuation.
    ax = axes[1, 1]
    ax.plot(tau, q_er, "o-", color=RED, label="Quantum")
    ax.plot(tau, c_er, "s--", color=NAVY, label="Classical")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Expansion duration $J\tau_{BC}$")
    ax.set_ylabel(r"$C$-specific $(E,R)$ fit residual")
    ax.set_xticks(tau, [str(int(x)) for x in tau])
    ax.legend(loc="best")
    finish_axes(ax)
    panel_label(ax, "(d)", 0.88, 0.94)

    return savefig(fig, "Fig4_robustness.pdf", h_pad=0.8, w_pad=0.8)


# =============================================================================
# Supplementary Figure S1: matched-preparation numerical validation
# =============================================================================


def make_figure_S1():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

    # (a) Target-adapted sampling convergence
    ax = axes[0]
    nbin = matched_conv["n_per_bin"].to_numpy(dtype=float)
    sm = matched_conv["sigma_BC_mean"].to_numpy(dtype=float)
    se = matched_conv["sigma_BC_se"].to_numpy(dtype=float)
    ax.errorbar(nbin, sm, yerr=se, fmt="s-", color=NAVY, capsize=2.5, label="Classical")
    ax.axhline(sigma_q_matched, color=RED, linewidth=1.5, label="Quantum")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"Samples per macro-bin")
    ax.set_ylabel(r"$\Sigma_{BC}/k_{\rm B}$")
    ax.legend()
    finish_axes(ax)
    panel_label(ax, "(a)", 0.02, 0.9)

    # (b) Deterministic quadrature convergence
    ax = axes[1]
    order = quad_conv["order"].to_numpy(dtype=float)
    SB = quad_conv["S_B"].to_numpy(dtype=float)
    SER = quad_conv["S_ER_C_representative"].to_numpy(dtype=float)
    ax.plot(order, SB, "o-", color=GREEN, label=r"$S_B^{\rm cl}$")
    ax.plot(order, SER, "s-", color=NAVY, label=r"$S_{E,R}^{\rm cl}(C)$")
    ax.set_xlabel(r"Gauss--Legendre order")
    ax.set_ylabel(r"Entropy$/k_{\rm B}$")
    ax.legend()
    finish_axes(ax)
    panel_label(ax, "(b)", 0.02, 0.9)

    return savefig(fig, "FigS1_matched_preparation_validation.pdf", w_pad=0.8)


# =============================================================================
# Supplementary Figure S2: time-step and QMC refinement
# =============================================================================


def make_figure_S2():
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2))

    # (a) protocol time-step refinement: percentage shift
    ax = axes[0, 0]
    ax.plot(
        protocol_dt["tau_exp_Jinv"],
        protocol_dt["relative_paired_shift_percent"],
        "o-", color=GREEN,
    )
    ax.axhline(0.0, linestyle=":", linewidth=1.2, color=GRAY)
    ax.set_xscale("log", base=2)
    ax.set_xticks(protocol_dt["tau_exp_Jinv"], [str(int(x)) for x in protocol_dt["tau_exp_Jinv"]])
    ax.set_xlabel(r"$J\tau_{\rm exp}$")
    ax.set_ylabel(r"$\Delta\Sigma_{BC}^{\rm cl}$ from $\Delta t$ refinement $(\%)$")
    finish_axes(ax)
    panel_label(ax, "(a)")

    # (b) protocol QMC-size refinement
    ax = axes[0, 1]
    ax.plot(
        protocol_n["tau_exp_Jinv"],
        protocol_n["relative_shift_percent"],
        "s-", color=CYAN,
    )
    ax.axhline(0.0, linestyle=":", linewidth=1.2, color=GRAY)
    ax.set_xscale("log", base=2)
    ax.set_xticks(protocol_n["tau_exp_Jinv"], [str(int(x)) for x in protocol_n["tau_exp_Jinv"]])
    ax.set_xlabel(r"$J\tau_{\rm exp}$")
    ax.set_ylabel(r"$2^{16}-2^{15}$ shift $(\%)$")
    finish_axes(ax)
    panel_label(ax, "(b)")

    # (c) L=6 sample-size refinement
    ax = axes[1, 0]
    ax.plot(l6_n["Q"], l6_n["relative_shift_percent"], "d-", color=NAVY)
    ax.axhline(0.0, linestyle=":", linewidth=1.2, color=GRAY)
    ax.set_xticks(l6_n["Q"].astype(int))
    ax.set_xlabel(r"Particle number $Q$")
    ax.set_ylabel(r"$2^{16}-2^{15}$ shift $(\%)$")
    finish_axes(ax)
    panel_label(ax, "(c)")

    # (d) L=6 time-step refinement
    ax = axes[1, 1]
    ax.plot(l6_dt["Q"], l6_dt["relative_shift_percent"], "^-", color=BROWN)
    ax.axhline(0.0, linestyle=":", linewidth=1.2, color=GRAY)
    ax.set_xticks(l6_dt["Q"].astype(int))
    ax.set_xlabel(r"Particle number $Q$")
    ax.set_ylabel(r"$\Delta\Sigma_{BC}^{\rm cl}$ from $\Delta t$ refinement $(\%)$")
    finish_axes(ax)
    panel_label(ax, "(d)")

    return savefig(fig, "FigS2_numerical_refinement.pdf", h_pad=0.8, w_pad=0.8)


# =============================================================================
# Numerical summary printed to notebook
# =============================================================================


def verify_selected_cycle_bookkeeping():
    """Audit the finalized representative Q=4 publication numbers.

    This is a retained-state/heat ledger.  It is intentionally distinct from
    run_exact_quantum_cycle_audit(), which reconstructs the microscopic quantum
    endpoint and independently integrates the explicit Hamiltonian paths.
    """
    T_h = 5.0
    T_c = 1.5

    S_B_q = 1.603917951433807
    S_C_q = 2.245527777382964
    S_D_q = 0.5554657988825167
    S_A_q = S_D_q
    Q_h_q = 5.242260762756452
    Q_c_q = 2.535092967750671
    audit_q = constructive_cycle_ledger(
        S_A=S_A_q,
        S_B=S_B_q,
        S_C=S_C_q,
        S_D=S_D_q,
        Q_h=Q_h_q,
        Q_c=Q_c_q,
        T_h=T_h,
        T_c=T_c,
    )

    S_B_cl = -2.0907880861013055
    S_C_cl = -1.3118599014778358
    S_D_cl = -3.139240238621622
    S_A_cl = S_D_cl
    Q_h_cl = 5.242260762601583
    Q_c_cl = 2.7410705057156792
    audit_cl = constructive_cycle_ledger(
        S_A=S_A_cl,
        S_B=S_B_cl,
        S_C=S_C_cl,
        S_D=S_D_cl,
        Q_h=Q_h_cl,
        Q_c=Q_c_cl,
        T_h=T_h,
        T_c=T_c,
    )

    tol = 5.0e-9
    for label, audit in (("quantum", audit_q), ("classical", audit_cl)):
        if abs(audit.sigma_AB_retained) > tol:
            raise RuntimeError(f"{label} A->B retained reversibility check failed.")
        if abs(audit.sigma_cold_isotherm_retained) > tol:
            raise RuntimeError(f"{label} Cbar->D retained reversibility check failed.")
        if abs(audit.sigma_DA_retained) > tol:
            raise RuntimeError(f"{label} D->A retained reversibility check failed.")
        if abs(audit.retained_balance_error) > tol:
            raise RuntimeError(f"{label} retained complete-cycle entropy balance failed.")
        if abs(audit.physical_balance_error) > tol:
            raise RuntimeError(f"{label} physical complete-cycle entropy balance failed.")
        if abs(audit.work_identity_error) > tol:
            raise RuntimeError(f"{label} complete-cycle work identity failed.")

    if not np.isclose(Q_h_q, Q_h_cl, rtol=0.0, atol=5.0e-9):
        raise RuntimeError("Representative same-Q_h matching failed.")

    return {"quantum": audit_q, "classical": audit_cl}


def print_key_numbers():
    cycle_audit = verify_selected_cycle_bookkeeping()
    record_term = float(matched_c["DeltaG_R"] - matched_q["DeltaG_R"])
    energy_term = float(-(matched_c["DeltaI_E"] - matched_q["DeltaI_E"]))
    total_gap = float(matched_c["Sigma_BC"] - matched_q["Sigma_BC"])

    print("\n=== FINAL COMPANION-PAPER PLOT INPUTS ===")
    print("Figure data source: embedded finalized publication values")
    print(f"Output folder: {OUTDIR}")
    print(f"Cycle completion: {CYCLE_CLOSURE_LABEL}")

    print("\nStrict matched-(p_B,E_B) Q=4 mechanism:")
    print(f"  Quantum Sigma_BC/kB   = {matched_q['Sigma_BC']:.9f}")
    print(f"  Classical Sigma_BC/kB = {matched_c['Sigma_BC']:.9f} +/- {matched_c['Sigma_BC_se']:.9f}")
    print(f"  Record contribution   = {record_term:+.9f}")
    print(f"  Energy contribution   = {energy_term:+.9f}")
    print(f"  Total gap              = {total_gap:+.9f}")
    print(f"  D[pC^Q||pi]            = {Dq_endpoint:.9f}")
    print(f"  D[pC^cl||pi]           = {Dc_endpoint:.9f}")

    aq = cycle_audit["quantum"]
    ac = cycle_audit["classical"]
    print("\nRepresentative Q=4 retained/physical cycle ledger:")
    print(f"  Quantum retained Sigma_BC/kB      = {aq.sigma_BC_retained:.9f}")
    print(f"  Quantum physical reset Sigma/kB   = {aq.sigma_reset_physical:.9f}")
    print(f"  Quantum retained Sigma_cyc/kB     = {aq.sigma_cycle_retained:.9f}")
    print(f"  Quantum physical Sigma_cyc/kB     = {aq.sigma_cycle_physical:.9f}")
    print(f"  Classical retained Sigma_BC/kB    = {ac.sigma_BC_retained:.9f}")
    print(f"  Classical physical reset Sigma/kB = {ac.sigma_reset_physical:.9f}")
    print(f"  Classical retained Sigma_cyc/kB   = {ac.sigma_cycle_retained:.9f}")
    print(f"  Classical physical Sigma_cyc/kB   = {ac.sigma_cycle_physical:.9f}")
    print("  The reset is the physical realization of Sigma_BC, not a second term.")

    if RUN_EXACT_QUANTUM_CYCLE_AUDIT:
        qa = run_exact_quantum_cycle_audit()
        print("\nExact constructive Q=4 quantum cycle audit:")
        print(f"  S_fine(C)/kB                    = {qa.S_fine_C:.12f}")
        print(f"  S_ER(C)/kB                      = {qa.S_ER_C:.12f}")
        print(f"  S_R(C)/kB                       = {qa.S_R_C:.12f}")
        print(f"  Sigma_BC/kB                     = {qa.sigma_BC:.12f}")
        print(f"  nested chi decomposition error  = {qa.nested_decomposition_error:.3e}")
        print(f"  max Tr(P_r chi_R) residual      = {qa.chi_R_record_residual:.3e}")
        print(f"  max Tr(P_r chi_ER) residual     = {qa.chi_ER_record_residual:.3e}")
        print(f"  Tr(H0 chi_ER) residual          = {qa.chi_ER_energy_residual:.3e}")
        print(f"  (E,R) fit residual              = {qa.ER_constraint_residual:.3e}")
        print(f"  ER-as-Gibbs residual            = {qa.ER_is_Gibbs_residual:.3e}")
        print(f"  H_CR energy identity error/J    = {qa.HCR_energy_identity_error:.3e}")
        print(f"  beta_C^* J                      = {qa.beta_C_star:.10f}")
        print(f"  alpha_C (alpha_4=0)             = {np.array(qa.alphas_C)}")
        print(f"  W_on(A->B)/J                    = {qa.W_AB_on:.12f}")
        print(f"  W_on(B->C^-)/J                  = {qa.W_BC_on:.12f}")
        print(f"  W_on(C^-->C^+)/J                = {qa.W_switch_on:.12f}")
        print(f"  W_on(C^+->Cbar)/J               = {qa.W_reset_on:.12f}")
        print(f"  Q_sys(C^+->Cbar)/J              = {qa.Q_reset_to_system:.3e}")
        print(f"  W_on(Cbar->D)/J                 = {qa.W_CbarD_on:.12f}")
        print(f"  W_on(D->A)/J                    = {qa.W_DA_on:.12f}")
        print(f"  Q_h/J                           = {qa.Q_h:.12f}")
        print(f"  Q_c/J                           = {qa.Q_c:.12f}")
        print(f"  W_net direct/J                  = {qa.W_net_direct:.12f}")
        print(f"  W_net from Q_h-Q_c/J            = {qa.W_net_from_heat:.12f}")
        print(f"  direct-work closure error/J     = {qa.work_closure_error:.3e}")
        print(f"  -Q_h/T_h+Q_c/T_c                = {qa.reservoir_entropy_change:.12f}")
        print(f"  entropy closure error           = {qa.entropy_closure_error:.3e}")
        print(f"  eta_Q                           = {qa.eta:.9f}")

    if RUN_FULL_CLASSICAL_CYCLE_AUDIT:
        ca = run_full_classical_cycle_audit(
            qmc_power=CLASSICAL_AUDIT_QMC_POWER,
            scrambles=CLASSICAL_AUDIT_SCRAMBLES,
            dt=CLASSICAL_AUDIT_DT,
            quadrature_order=CLASSICAL_AUDIT_QUADRATURE_ORDER,
        )
        print("\nFresh classical microscopic closure audit:")
        print("  (This is a new QMC rerun, so it fluctuates around the embedded production means.)")
        print(f"  Sigma_BC/kB                     = {ca.sigma_BC:.9f}")
        print(f"  e_C/J                           = {ca.endpoint_energy_per_particle:.9f} +/- {ca.endpoint_energy_se:.3e}")
        print(f"  p_C                             = {np.array(ca.endpoint_probabilities)}")
        print(f"  max p_C standard error          = {ca.endpoint_probability_se_max:.3e}")
        print(f"  beta_C^* J                      = {ca.beta_C_star:.10f}")
        print(f"  alpha_C (alpha_4=0)             = {np.array(ca.alphas_C)}")
        print(f"  ER energy residual              = {ca.ER_energy_residual:.3e}")
        print(f"  H_CR energy identity error/J    = {ca.HCR_energy_identity_error:.3e}")
        print(f"  Q_sys(C^+->Cbar)/J              = {ca.Q_reset_to_system:.3e}")
        print(f"  W_switch/J                      = {ca.W_switch_on:.9f}")
        print(f"  W_net direct/J                  = {ca.W_net_direct:.9f}")
        print(f"  W_net from heat/J               = {ca.W_net_from_heat:.9f}")
        print(f"  work closure error/J            = {ca.work_closure_error:.3e}")
        print(f"  entropy closure error           = {ca.entropy_closure_error:.3e}")

    print("\nProtocol-duration endpoint reconstruction audit:")
    for _, row in protocol.iterrows():
        print(
            f"  J tau_BC={int(row['tau_exp_Jinv']):>2d}: "
            f"quantum ER residual={row['quantum_ER_residual']:.3e}, "
            f"classical ER residual={row['classical_ER_max_residual']:.3e}; "
            "H_CR is rebuilt for this endpoint"
        )

    print("\nQ-scaling complete-cycle endpoints:")
    for _, row in qscale.iterrows():
        print(
            f"  Q={int(row['Q'])}: Sigma_cyc^Q={row['sigma_cyc_q']:.6f}, "
            f"Sigma_cyc^cl={row['sigma_cyc_cl']:.6f}, "
            f"gap={row['sigma_cyc_gap']:.6f}, "
            f"eta advantage={row['eta_adv_pp']:.3f} pp"
        )


# =============================================================================
# Execute
# =============================================================================


print_key_numbers()

if MAKE_MAIN_FIGURES:
    make_figure_1()
    make_figure_2()
    make_figure_3()
    make_figure_4()

if MAKE_SUPPLEMENTARY:
    make_figure_S1()
    make_figure_S2()

print("\nAll requested figures generated.")
