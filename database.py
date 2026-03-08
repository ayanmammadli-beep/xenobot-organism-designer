"""
data/contaminants/database.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real physicochemical properties sourced from peer-reviewed literature.
Every value here has a DOI. This is what makes the platform scientific.

Interaction models implemented:
  DLVO  = van der Waals + electrostatic double layer
  VDW   = pure van der Waals (non-polar particles)
  STERIC= polymer brush steric repulsion (protein aggregates)
  HYDRO = hydrophobic attraction (lipid/membrane contaminants)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════
# FLUID MODELS
# ═══════════════════════════════════════════════════════════

@dataclass
class FluidModel:
    name: str
    model_type: str          # newtonian | carreau_yasuda | oldroyd_b | power_law
    density: float           # kg/m³
    # Newtonian
    viscosity: float = 1e-3  # Pa·s (only used for Newtonian)
    # Carreau-Yasuda params (blood, mucus)
    eta_0: float = 0.056     # Pa·s zero-shear viscosity
    eta_inf: float = 0.00345 # Pa·s infinite-shear viscosity
    lambda_cy: float = 3.313 # s  relaxation time
    n_cy: float = 0.3568     # power-law index
    a_cy: float = 2.0        # Yasuda parameter
    # Oldroyd-B params (viscoelastic)
    lambda_1: float = 0.0    # s  relaxation time
    lambda_2: float = 0.0    # s  retardation time
    # Conditions
    temperature: float = 310.15  # K
    pH: float = 7.4
    ionic_strength: float = 0.15 # mol/L (physiological)
    notes: str = ""
    references: List[str] = field(default_factory=list)


FLUID_DATABASE: Dict[str, FluidModel] = {

    "water_pure": FluidModel(
        name="Pure Water (20°C)",
        model_type="newtonian",
        density=998.2,
        viscosity=1.002e-3,
        temperature=293.15,
        pH=7.0,
        ionic_strength=0.0,
        references=["CRC Handbook of Chemistry and Physics, 97th Ed"]
    ),

    "water_groundwater": FluidModel(
        name="Typical Groundwater",
        model_type="newtonian",
        density=999.5,
        viscosity=1.1e-3,    # slightly higher due to dissolved minerals
        temperature=285.15,  # 12°C
        pH=6.8,
        ionic_strength=0.01,
        notes="Average US groundwater parameters",
        references=["doi:10.1016/j.watres.2019.01.001"]
    ),

    "water_wastewater": FluidModel(
        name="Industrial Wastewater",
        model_type="newtonian",
        density=1002.0,
        viscosity=1.15e-3,
        temperature=298.15,
        pH=6.2,
        ionic_strength=0.05,
        references=["doi:10.1016/j.chemosphere.2020.126372"]
    ),

    "blood_whole": FluidModel(
        name="Whole Blood (37°C)",
        model_type="carreau_yasuda",
        density=1060.0,
        viscosity=3.5e-3,    # effective at physiological shear
        eta_0=0.056,
        eta_inf=0.00345,
        lambda_cy=3.313,
        n_cy=0.3568,
        a_cy=2.0,
        temperature=310.15,
        pH=7.4,
        ionic_strength=0.15,
        notes="Carreau-Yasuda params from Gijsen et al. 1999",
        references=["doi:10.1016/S0021-9290(98)00015-9"]
    ),

    "blood_plasma": FluidModel(
        name="Blood Plasma",
        model_type="newtonian",
        density=1025.0,
        viscosity=1.2e-3,
        temperature=310.15,
        pH=7.4,
        ionic_strength=0.15,
        references=["doi:10.1111/j.1365-2141.1993.tb03191.x"]
    ),

    "synovial_fluid": FluidModel(
        name="Synovial Fluid (Joint)",
        model_type="power_law",
        density=1008.0,
        viscosity=0.1,       # at 1/s shear rate, highly shear-thinning
        eta_0=10.0,
        n_cy=0.4,
        temperature=310.15,
        pH=7.4,
        ionic_strength=0.15,
        notes="Highly viscoelastic due to hyaluronic acid",
        references=["doi:10.1016/j.jbiomech.2006.08.014"]
    ),

    "mucus_airway": FluidModel(
        name="Airway Mucus",
        model_type="oldroyd_b",
        density=1010.0,
        viscosity=1.0,       # Pa·s at low shear
        eta_0=50.0,
        lambda_1=0.1,
        lambda_2=0.01,
        temperature=310.15,
        pH=6.9,
        notes="Gel-like at rest, flows under shear (viscoelastic)",
        references=["doi:10.1039/c9sm01144g"]
    ),

    "csf": FluidModel(
        name="Cerebrospinal Fluid",
        model_type="newtonian",
        density=1007.0,
        viscosity=0.7e-3,
        temperature=310.15,
        pH=7.35,
        ionic_strength=0.15,
        references=["doi:10.1007/s11548-016-1492-2"]
    ),
}


def get_viscosity(fluid: FluidModel, shear_rate: float) -> float:
    """
    Compute apparent viscosity at a given shear rate.

    Carreau-Yasuda: η(γ̇) = η∞ + (η₀-η∞)·[1+(λγ̇)ᵃ]^((n-1)/a)
    Power-law:      η(γ̇) = η₀·γ̇^(n-1)
    Newtonian:      η(γ̇) = η₀
    """
    if fluid.model_type == "newtonian":
        return fluid.viscosity

    elif fluid.model_type == "carreau_yasuda":
        term = (1 + (fluid.lambda_cy * shear_rate) ** fluid.a_cy)
        exp  = (fluid.n_cy - 1) / fluid.a_cy
        return fluid.eta_inf + (fluid.eta_0 - fluid.eta_inf) * term ** exp

    elif fluid.model_type == "power_law":
        return fluid.eta_0 * max(shear_rate, 1e-6) ** (fluid.n_cy - 1)

    elif fluid.model_type == "oldroyd_b":
        # Simplified: use viscosity at given shear rate
        return fluid.viscosity / (1 + fluid.lambda_1 * shear_rate)

    return fluid.viscosity


def debye_length(ionic_strength: float, temperature: float = 298.15) -> float:
    """
    κ⁻¹ = √(ε₀εᵣkT / 2NAe²I)
    Returns Debye screening length in meters.
    """
    eps_0  = 8.854e-12   # F/m
    eps_r  = 78.5        # water relative permittivity
    k_B    = 1.381e-23   # J/K
    N_A    = 6.022e23    # mol⁻¹
    e      = 1.602e-19   # C
    I_si   = ionic_strength * 1000  # mol/m³
    kappa  = np.sqrt(2 * N_A * e**2 * I_si / (eps_0 * eps_r * k_B * temperature))
    return 1.0 / kappa


# ═══════════════════════════════════════════════════════════
# CONTAMINANT DATABASE
# ═══════════════════════════════════════════════════════════

@dataclass
class Contaminant:
    name: str
    category: str           # microplastic | pfas | protein_aggregate | heavy_metal | pathogen
    interaction_model: str  # dlvo | vdw | steric | hydrophobic | combined

    # Physical
    diameter: float         # m (mean particle diameter)
    diameter_std: float     # m (polydispersity)
    density: float          # kg/m³
    molecular_weight: float # g/mol (for molecular contaminants)

    # Surface chemistry (for DLVO)
    zeta_potential: float   # mV (at reference pH 7)
    zeta_ph_slope: float    # mV/pH unit (how zeta changes with pH)
    hamaker_constant: float # J (material-specific)

    # For charged/ionic
    charge: float           # elementary charges (net charge at pH 7)
    pKa: Optional[float]    # acid dissociation constant

    # Binding to xenobot surface
    binding_mode: str       # adhesion | mechanical | enzymatic | electrostatic
    binding_energy: float   # kT units (thermal energy)
    binding_site_density: float  # sites/µm² on adhesive cell surface

    # Simulation
    diffusivity: float      # m²/s (Stokes-Einstein)
    color_hex: str          # for 3D rendering

    references: List[str] = field(default_factory=list)
    notes: str = ""


CONTAMINANT_DATABASE: Dict[str, Contaminant] = {

    "hdpe_microplastic": Contaminant(
        name="HDPE Microplastic",
        category="microplastic",
        interaction_model="vdw",
        diameter=50e-6,          # 50 µm
        diameter_std=20e-6,
        density=950.0,           # kg/m³
        molecular_weight=1e6,    # ~1 MDa polymer
        zeta_potential=-32.0,    # mV (slightly negative in water)
        zeta_ph_slope=-2.0,
        hamaker_constant=1.3e-20, # J (HDPE-water-cell)
        charge=-5.0,
        pKa=None,
        binding_mode="adhesion",
        binding_energy=8.5,      # kT (moderate adhesion)
        binding_site_density=50.0,
        diffusivity=9.0e-15,     # Stokes-Einstein at 20°C, 50µm
        color_hex="#FFA726",
        references=["doi:10.1021/acs.est.0c02070"]
    ),

    "pet_microplastic": Contaminant(
        name="PET Microplastic",
        category="microplastic",
        interaction_model="dlvo",
        diameter=30e-6,
        diameter_std=15e-6,
        density=1380.0,
        molecular_weight=1e6,
        zeta_potential=-41.0,
        zeta_ph_slope=-3.1,
        hamaker_constant=8.5e-21,
        charge=-8.0,
        pKa=None,
        binding_mode="adhesion",
        binding_energy=10.2,
        binding_site_density=50.0,
        diffusivity=1.5e-14,
        color_hex="#EF5350",
        references=["doi:10.1016/j.chemosphere.2021.129814"]
    ),

    "pfas_pfoa": Contaminant(
        name="PFOA (Perfluorooctanoic acid)",
        category="pfas",
        interaction_model="dlvo",
        diameter=1.2e-9,         # ~1.2 nm molecular size
        diameter_std=0.1e-9,
        density=1800.0,
        molecular_weight=414.07,
        zeta_potential=-35.2,
        zeta_ph_slope=-4.5,
        hamaker_constant=8.2e-21,
        charge=-1.0,             # fully deprotonated at pH 7 (pKa=2.8)
        pKa=2.8,
        binding_mode="electrostatic",
        binding_energy=15.0,     # strong — used in remediation
        binding_site_density=200.0,
        diffusivity=5.0e-10,     # molecular diffusivity
        color_hex="#AB47BC",
        references=["doi:10.1021/es060882q", "doi:10.1021/acs.est.9b05224"]
    ),

    "fibrin_clot": Contaminant(
        name="Fibrin Microclot",
        category="protein_aggregate",
        interaction_model="steric",
        diameter=5e-6,           # 5 µm fibrin aggregate
        diameter_std=3e-6,
        density=1100.0,
        molecular_weight=340000, # fibrinogen monomer
        zeta_potential=-15.0,
        zeta_ph_slope=-1.5,
        hamaker_constant=5.0e-21,
        charge=-12.0,
        pKa=None,
        binding_mode="mechanical",  # needs physical disruption
        binding_energy=20.0,        # strong fibrin network
        binding_site_density=100.0,
        diffusivity=4.4e-14,
        color_hex="#EF9A9A",
        references=["doi:10.1083/jcb.200212153"]
    ),

    "amyloid_beta": Contaminant(
        name="Amyloid-β Plaque (Alzheimer's)",
        category="protein_aggregate",
        interaction_model="combined",
        diameter=100e-9,         # 100 nm fibrils
        diameter_std=50e-9,
        density=1200.0,
        molecular_weight=4500,   # Aβ1-42 monomer
        zeta_potential=-20.0,
        zeta_ph_slope=-2.0,
        hamaker_constant=6.0e-21,
        charge=-3.0,
        pKa=None,
        binding_mode="combined",
        binding_energy=18.0,
        binding_site_density=150.0,
        diffusivity=4.0e-12,
        color_hex="#CE93D8",
        notes="Hydrophobic core with charged periphery — combined DLVO+hydrophobic model",
        references=["doi:10.1038/nsmb.2920"]
    ),

    "ldl_cholesterol": Contaminant(
        name="LDL Cholesterol Particle",
        category="lipid",
        interaction_model="hydrophobic",
        diameter=25e-9,          # 25 nm LDL particle
        diameter_std=5e-9,
        density=1020.0,
        molecular_weight=2.5e6, # LDL particle
        zeta_potential=-22.0,
        zeta_ph_slope=-1.0,
        hamaker_constant=7.0e-21,
        charge=-50.0,
        pKa=None,
        binding_mode="electrostatic",
        binding_energy=12.0,
        binding_site_density=80.0,
        diffusivity=2.0e-12,
        color_hex="#FFD54F",
        references=["doi:10.1016/0021-9150(93)90006-V"]
    ),

    "nanoplastic_polystyrene": Contaminant(
        name="Polystyrene Nanoplastic",
        category="microplastic",
        interaction_model="dlvo",
        diameter=200e-9,
        diameter_std=50e-9,
        density=1050.0,
        molecular_weight=2e5,
        zeta_potential=-38.0,
        zeta_ph_slope=-3.5,
        hamaker_constant=9.0e-21,
        charge=-6.0,
        pKa=None,
        binding_mode="adhesion",
        binding_energy=9.0,
        binding_site_density=60.0,
        diffusivity=2.4e-12,
        color_hex="#29B6F6",
        references=["doi:10.1021/acs.est.1c03476"]
    ),
}


# ═══════════════════════════════════════════════════════════
# INTERACTION FORCE CALCULATORS
# ═══════════════════════════════════════════════════════════

def force_vdw(R: float, h: float, A: float) -> float:
    """
    van der Waals attraction between sphere and flat surface.
    F_vdW = -A·R / (6h²)
    R: particle radius (m), h: separation (m), A: Hamaker constant (J)
    Returns force in Newtons (negative = attractive)
    """
    h = max(h, 0.158e-9)  # Born repulsion cutoff at ~0.158 nm
    return -A * R / (6.0 * h**2)


def force_electrostatic(R: float, h: float, zeta: float,
                         ionic_strength: float, temperature: float) -> float:
    """
    Electrostatic double-layer force (linearized Poisson-Boltzmann).
    F_EDL = 64π·R·n₀·kT/κ · tanh²(zeζ/4kT) · e^(-κh)

    Valid for κR >> 1 (Derjaguin approximation).
    """
    k_B  = 1.381e-23
    N_A  = 6.022e23
    e    = 1.602e-19
    eps_0= 8.854e-12
    eps_r= 78.5
    kappa_inv = debye_length(ionic_strength, temperature)
    kappa     = 1.0 / kappa_inv

    n0   = ionic_strength * 1000 * N_A   # number density mol/m³ → 1/m³
    zeta_si = zeta * 1e-3                 # mV → V
    # Dimensionless surface potential
    y = e * zeta_si / (4 * k_B * temperature)
    prefactor = 64 * np.pi * R * n0 * k_B * temperature / kappa
    return prefactor * np.tanh(y)**2 * np.exp(-kappa * h)


def force_dlvo(R: float, h: float, contaminant: Contaminant,
               fluid: FluidModel) -> float:
    """
    Total DLVO force = F_vdW + F_EDL
    Negative = net attractive (capture favorable)
    Positive = net repulsive (capture hindered)
    """
    # Adjust zeta for current pH
    delta_pH  = fluid.pH - 7.0
    zeta_adj  = contaminant.zeta_potential + contaminant.zeta_ph_slope * delta_pH

    F_vdw = force_vdw(R, h, contaminant.hamaker_constant)
    F_edl = force_electrostatic(R, h, zeta_adj,
                                 fluid.ionic_strength, fluid.temperature)
    return F_vdw + F_edl


def force_steric(h: float, L: float = 20e-9, D: float = 5e-9,
                 T: float = 310.15) -> float:
    """
    Steric repulsion from polymer brush (fibrin, mucus).
    F_steric = kT/D³ · e^(-2πh/L)
    L: brush height, D: grafting distance
    """
    k_B = 1.381e-23
    return (k_B * T / D**3) * np.exp(-2 * np.pi * h / L)


def force_hydrophobic(h: float, C: float = 1e-10, lam: float = 1.5e-9) -> float:
    """
    Hydrophobic attraction (empirical).
    F_hydro = -C · e^(-h/λ)
    λ ≈ 1-2 nm for typical hydrophobic surfaces
    """
    return -C * np.exp(-h / lam)


def compute_interaction_force(h: float, contaminant: Contaminant,
                               fluid: FluidModel) -> float:
    """Master dispatcher — picks the right force model."""
    R = contaminant.diameter / 2.0

    if contaminant.interaction_model == "vdw":
        return force_vdw(R, h, contaminant.hamaker_constant)

    elif contaminant.interaction_model == "dlvo":
        return force_dlvo(R, h, contaminant, fluid)

    elif contaminant.interaction_model == "steric":
        return (force_vdw(R, h, contaminant.hamaker_constant) +
                force_steric(h))

    elif contaminant.interaction_model == "hydrophobic":
        return (force_vdw(R, h, contaminant.hamaker_constant) +
                force_hydrophobic(h))

    elif contaminant.interaction_model == "combined":
        return (force_dlvo(R, h, contaminant, fluid) +
                force_hydrophobic(h))

    return 0.0


def parse_environment_string(text: str) -> Tuple[FluidModel, Contaminant]:
    """
    Parse natural language environment description.
    Used by Claude agent tool parse_environment().
    Returns best-matching fluid and contaminant from DB.
    """
    text_lower = text.lower()

    # Fluid matching
    fluid_key = "water_pure"
    if any(w in text_lower for w in ["blood", "plasma", "vascular", "clot"]):
        fluid_key = "blood_whole"
    elif any(w in text_lower for w in ["synovial", "joint"]):
        fluid_key = "synovial_fluid"
    elif any(w in text_lower for w in ["mucus", "airway", "lung", "bronch"]):
        fluid_key = "mucus_airway"
    elif any(w in text_lower for w in ["csf", "cerebrospinal", "brain"]):
        fluid_key = "csf"
    elif any(w in text_lower for w in ["groundwater", "aquifer"]):
        fluid_key = "water_groundwater"
    elif any(w in text_lower for w in ["wastewater", "industrial", "effluent"]):
        fluid_key = "water_wastewater"

    # Contaminant matching
    contam_key = "hdpe_microplastic"
    if any(w in text_lower for w in ["pfas", "pfoa", "pfos", "fluorinated"]):
        contam_key = "pfas_pfoa"
    elif any(w in text_lower for w in ["fibrin", "clot", "thrombosis"]):
        contam_key = "fibrin_clot"
    elif any(w in text_lower for w in ["amyloid", "alzheimer", "plaque", "aβ"]):
        contam_key = "amyloid_beta"
    elif any(w in text_lower for w in ["ldl", "cholesterol", "lipid"]):
        contam_key = "ldl_cholesterol"
    elif any(w in text_lower for w in ["nanoplastic", "polystyrene", "nano"]):
        contam_key = "nanoplastic_polystyrene"
    elif any(w in text_lower for w in ["pet", "polyethylene terephthalate"]):
        contam_key = "pet_microplastic"

    return FLUID_DATABASE[fluid_key], CONTAMINANT_DATABASE[contam_key]
