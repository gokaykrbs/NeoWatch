"""
NeoWatch - Orbital Playground Physics Engine & Earth Impact Effects Simulator
=============================================================================
Implementation of the Earth Impact Effects & Kinetic Simulator v1.0 specifications
defined in 'Playground Fizik Motoru Spesifikasyonları.pdf' / Orbital Playground Spec.

Core Equations & Features:
1. Kinetic Energy & TNT Equivalent Yield:
   - Ek = 0.5 * m * v^2 (Joules)
   - Emegaton = Ek / (4.184 * 10^15) (Megatons of TNT)
2. Pi-Scaling Transient Crater Diameter (Pi-Scaling Law):
   - Dtc = 1.161 * (rho_i / rho_t)^(1/3) * Di^0.78 * v^0.44 * g^(-0.22) * sin^(1/3)(theta)
   - Atmospheric Skip Detection: If theta < 10 degrees, trigger skip warning and stop cratering calculation.
3. Damage Report Classification Matrix:
   - < 10 Mt: Airburst (Upper Atmosphere Inflight Detonation)
   - 10 - 100 Mt: Local Destruction (City-Scale Devastation)
   - 100 - 1,000,000 Mt: Regional Devastation (Continental Impact & Major Crater)
   - > 1,000,000 Mt: Global Extinction Threat (Impact Winter & Biospheric Collapse)
4. Extended Impact Telemetry:
   - Seismic equivalent (Richter Magnitude Mw)
   - Thermal radiation fireball radius
   - Shockwave airblast radii (20 psi, 5 psi, 1 psi)
   - Final crater dimensions (diameter and depth)
   - Equivalent benchmark comparisons (Hiroshima, Tunguska, Chicxulub)
5. 3D Plotly Earth & Trajectory Visualizers:
   - 3D Interactive Globe with spherical coordinates
   - Hypersonic entry trajectory & skip trajectory modeling
   - Concentric surface shockwave ripple circles (Pi-crater, 20psi, 5psi, 1psi)
   - Energetics and blast radius visualization
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import plotly.graph_objects as go


# Standard Physical Constants
EARTH_GRAVITY_G = 9.81                 # m/s^2
EARTH_RADIUS_KM = 6371.0               # Mean Earth radius in km
DEFAULT_TARGET_DENSITY_RHO_T = 2500.0  # kg/m^3 (Earth's crust / sedimentary-crystalline target)
DEFAULT_ASTEROID_DENSITY_RHO_I = 3000.0# kg/m^3 (Standard stony meteorite)
JOULES_PER_MEGATON_TNT = 4.184e15      # Joules in 1 Megaton TNT
HIROSHIMA_ENERGY_JOULES = 6.276e13     # ~15 kilotons TNT (6.276 * 10^13 J)
TUNGUSKA_ENERGY_MEGATONS = 15.0        # ~15 Mt TNT
CHICXULUB_ENERGY_MEGATONS = 1.0e8      # ~100 Million Mt TNT


# Preset Material Densities (kg/m^3)
DENSITY_PRESETS = {
    "Stony Meteorite (Chondrite) [3,000 kg/m³]": 3000.0,
    "Iron-Nickel Meteorite (Dense Metal) [7,800 kg/m³]": 7800.0,
    "Carbonaceous Asteroid (Porous Rock) [2,200 kg/m³]": 2200.0,
    "Cometary Nucleus (Ice / Porous Dust) [1,000 kg/m³]": 1000.0,
    "Dense Basaltic Asteroid [3,500 kg/m³]": 3500.0,
}

# Historical and Famous Asteroid Presets
ASTEROID_PRESETS = {
    "99942 Apophis (2029 Close Approach)": {
        "diameter_m": 370.0,
        "velocity_km_s": 30.7,
        "density_kg_m3": 3200.0,
        "angle_deg": 45.0,
        "target": "Pacific Ocean (Sub-equatorial)",
        "lat": 14.5,
        "lon": -135.0,
        "desc": "Stony (LL-chondrite) NEO discovered in 2004. Passing within ~31,600 km of Earth in April 2029.",
    },
    "101955 Bennu (OSIRIS-REx Target)": {
        "diameter_m": 490.0,
        "velocity_km_s": 27.8,
        "density_kg_m3": 1190.0,
        "angle_deg": 45.0,
        "target": "Atlantic Ocean (Mid-Latitude)",
        "lat": 28.0,
        "lon": -40.0,
        "desc": "Carbonaceous B-type rubble-pile asteroid sampled by NASA OSIRIS-REx mission.",
    },
    "Tunguska Event Impact Body (1908)": {
        "diameter_m": 60.0,
        "velocity_km_s": 20.0,
        "density_kg_m3": 2500.0,
        "angle_deg": 35.0,
        "target": "Siberian Taiga, Russia",
        "lat": 60.9,
        "lon": 101.9,
        "desc": "Stony asteroid / cometary airburst that flattened ~2,150 km² of Siberian forest.",
    },
    "Chelyabinsk Superbolide (2013)": {
        "diameter_m": 19.8,
        "velocity_km_s": 19.2,
        "density_kg_m3": 3300.0,
        "angle_deg": 18.3,
        "target": "Chelyabinsk Oblast, Russia",
        "lat": 54.8,
        "lon": 61.1,
        "desc": "LL5 chondrite airburst at ~29.7 km altitude, yielding ~500 kt TNT equivalent shockwave.",
    },
    "Chicxulub Dinosaur Extinction Impactor (~66 Ma)": {
        "diameter_m": 12000.0,  # 12 km
        "velocity_km_s": 20.0,
        "density_kg_m3": 2600.0,
        "angle_deg": 60.0,
        "target": "Yucatan Peninsula, Mexico",
        "lat": 21.3,
        "lon": -89.5,
        "desc": "Cretaceous-Paleogene extinction impactor creating ~180 km wide crater and global winter.",
    },
    "Barringer Meteor Crater Impactor (Canyon Diablo)": {
        "diameter_m": 50.0,
        "velocity_km_s": 12.8,
        "density_kg_m3": 7800.0,
        "angle_deg": 45.0,
        "target": "Arizona, USA",
        "lat": 35.0,
        "lon": -111.0,
        "desc": "Nickel-iron meteorite that created the 1.2 km wide Barringer Crater in Arizona ~50,000 years ago.",
    },
}

TARGET_LOCATIONS = {
    "Pacific Ocean (Deep Water)": {"lat": 14.5, "lon": -135.0},
    "Atlantic Ocean (Mid-Atlantic)": {"lat": 28.0, "lon": -40.0},
    "North America (Midwest USA)": {"lat": 39.0, "lon": -98.0},
    "Central Europe (Alps Basin)": {"lat": 46.8, "lon": 8.2},
    "Eurasian Taiga (Siberia)": {"lat": 60.9, "lon": 101.9},
    "Sahara Desert (North Africa)": {"lat": 23.4, "lon": 12.5},
    "Indian Ocean": {"lat": -15.0, "lon": 80.0},
    "Custom Coordinates": {"lat": 0.0, "lon": 0.0},
}


@dataclass
class ImpactParameters:
    """Input parameters for asteroid impact physics simulation."""
    diameter_m: float
    velocity_m_s: float
    angle_deg: float
    density_asteroid_kg_m3: float = DEFAULT_ASTEROID_DENSITY_RHO_I
    density_target_kg_m3: float = DEFAULT_TARGET_DENSITY_RHO_T
    mass_kg: Optional[float] = None
    gravity_m_s2: float = EARTH_GRAVITY_G


@dataclass
class ImpactResults:
    """Simulation results produced by the physics engine."""
    is_atmospheric_skip: bool
    skip_warning_message: Optional[str]
    diameter_m: float
    velocity_km_s: float
    angle_deg: float
    density_asteroid_kg_m3: float
    density_target_kg_m3: float
    mass_kg: float
    mass_metric_tons: float
    kinetic_energy_joules: float
    kinetic_energy_megatons: float
    hiroshima_bombs_equivalent: float
    tunguska_equivalent: float
    chicxulub_equivalent: float
    transient_crater_diameter_m: float
    transient_crater_diameter_km: float
    final_crater_diameter_m: float
    final_crater_diameter_km: float
    crater_depth_m: float
    classification: str
    expected_environmental_effect: str
    severity_level: str
    theme_color: str
    richter_magnitude: float
    fireball_radius_km: float
    blast_radius_20psi_heavy_km: float
    blast_radius_5psi_moderate_km: float
    blast_radius_1psi_light_km: float
    airburst_altitude_km: Optional[float]


class ImpactPhysicsEngine:
    """
    High-precision physics engine for celestial kinetic impact simulation,
    energy partition, Pi-scaling crater mechanics, and atmospheric entry dynamics.
    """

    @staticmethod
    def calculate_mass(diameter_m: float, density_kg_m3: float) -> float:
        radius_m = diameter_m / 2.0
        volume_m3 = (4.0 / 3.0) * math.pi * (radius_m ** 3)
        return volume_m3 * density_kg_m3

    @classmethod
    def simulate(cls, params: ImpactParameters) -> ImpactResults:
        diameter_m = float(params.diameter_m)
        velocity_m_s = float(params.velocity_m_s)
        angle_deg = float(params.angle_deg)
        density_i = float(params.density_asteroid_kg_m3)
        density_t = float(params.density_target_kg_m3)
        g = float(params.gravity_m_s2)

        if params.mass_kg is not None and params.mass_kg > 0:
            mass_kg = float(params.mass_kg)
        else:
            mass_kg = cls.calculate_mass(diameter_m, density_i)

        mass_metric_tons = mass_kg / 1000.0
        velocity_km_s = velocity_m_s / 1000.0

        # Kinetic Energy (Ek) & Megaton TNT Yield (Emegaton)
        kinetic_energy_joules = 0.5 * mass_kg * (velocity_m_s ** 2)
        kinetic_energy_megatons = kinetic_energy_joules / JOULES_PER_MEGATON_TNT

        hiroshima_equiv = kinetic_energy_joules / HIROSHIMA_ENERGY_JOULES
        tunguska_equiv = kinetic_energy_megatons / TUNGUSKA_ENERGY_MEGATONS
        chicxulub_equiv = kinetic_energy_megatons / CHICXULUB_ENERGY_MEGATONS

        # Atmospheric Skip Check: theta < 10 degrees
        is_skip = angle_deg < 10.0
        if is_skip:
            return ImpactResults(
                is_atmospheric_skip=True,
                skip_warning_message="⚠️ Atmospheric Skip Occurred: The entry angle is below 10°, causing the asteroid to graze Earth's upper atmosphere (~85 km) and ricochet back into deep space. Ground cratering and surface kinetic damage calculations are suspended.",
                diameter_m=diameter_m,
                velocity_km_s=velocity_km_s,
                angle_deg=angle_deg,
                density_asteroid_kg_m3=density_i,
                density_target_kg_m3=density_t,
                mass_kg=mass_kg,
                mass_metric_tons=mass_metric_tons,
                kinetic_energy_joules=kinetic_energy_joules,
                kinetic_energy_megatons=kinetic_energy_megatons,
                hiroshima_bombs_equivalent=hiroshima_equiv,
                tunguska_equivalent=tunguska_equiv,
                chicxulub_equivalent=chicxulub_equiv,
                transient_crater_diameter_m=0.0,
                transient_crater_diameter_km=0.0,
                final_crater_diameter_m=0.0,
                final_crater_diameter_km=0.0,
                crater_depth_m=0.0,
                classification="Atmospheric Skip (< 10°)",
                expected_environmental_effect="Asteroid entry angle is below the critical grazing threshold (10°). The impactor ricochets off the upper atmospheric boundary back into interplanetary orbit without ground impact or cratering.",
                severity_level="low",
                theme_color="#38bdf8",
                richter_magnitude=0.0,
                fireball_radius_km=0.0,
                blast_radius_20psi_heavy_km=0.0,
                blast_radius_5psi_moderate_km=0.0,
                blast_radius_1psi_light_km=0.0,
                airburst_altitude_km=85.0,
            )

        # Pi-Scaling Law for Transient Crater Diameter
        theta_rad = math.radians(angle_deg)
        density_ratio_term = (density_i / density_t) ** (1.0 / 3.0)
        diameter_term = diameter_m ** 0.78
        velocity_term = velocity_m_s ** 0.44
        gravity_term = g ** (-0.22)
        angle_term = (math.sin(theta_rad)) ** (1.0 / 3.0)

        d_tc_m = 1.161 * density_ratio_term * diameter_term * velocity_term * gravity_term * angle_term
        d_tc_km = d_tc_m / 1000.0

        # Simple to Complex crater transition
        if d_tc_km < 3.2:
            d_final_km = 1.25 * d_tc_km
            crater_depth_m = d_final_km * 1000.0 / 5.0
        else:
            d_final_km = 1.17 * (d_tc_km ** 1.13) / (3.2 ** 0.13)
            crater_depth_m = min(d_final_km * 1000.0 / 7.0, 3500.0)

        d_final_m = d_final_km * 1000.0

        # Damage Classification Matrix
        if kinetic_energy_megatons < 10.0:
            classification = "Airburst (< 10 Mt)"
            effect = "Detonates in the upper/mid atmosphere prior to surface impact (e.g., Chelyabinsk event). Generates strong airburst shockwaves capable of shattering windows and causing light structural damage."
            severity_level = "moderate" if kinetic_energy_megatons > 1.0 else "low"
            theme_color = "#f59e0b" if kinetic_energy_megatons > 1.0 else "#10b981"
            airburst_alt = max(5.0, 35.0 - (kinetic_energy_megatons * 2.0))
        elif 10.0 <= kinetic_energy_megatons < 100.0:
            classification = "Local Destruction (10 - 100 Mt)"
            effect = "Causes city-scale devastation. Severe blast overpressure flattens forests and structures over hundreds of square kilometers (e.g., 1908 Tunguska event) with localized thermal firestorms."
            severity_level = "severe"
            theme_color = "#f97316"
            airburst_alt = max(0.0, 10.0 - (kinetic_energy_megatons / 10.0))
        elif 100.0 <= kinetic_energy_megatons <= 1_000_000.0:
            classification = "Regional Devastation (100 - 1,000,000 Mt)"
            effect = "Massive crater excavation, severe regional earthquakes (Richter 7.0–9.0+), continent-wide thermal radiation, regional climate disruption, and catastrophic wildfire ignition."
            severity_level = "extreme"
            theme_color = "#ef4444"
            airburst_alt = 0.0
        else:
            classification = "Global Extinction Threat (> 1,000,000 Mt)"
            effect = "Ejects vast dust and sulfate aerosol plumes into the stratosphere, triggering a 'Global Impact Winter', collapsing global photosynthesis, and threatening biospheric mass extinction (Chicxulub-scale)."
            severity_level = "cataclysmic"
            theme_color = "#a855f7"
            airburst_alt = 0.0

        if kinetic_energy_joules > 1e10:
            richter_mag = round(max(0.0, 0.67 * math.log10(kinetic_energy_joules) - 5.87), 1)
        else:
            richter_mag = 0.0

        fireball_radius_km = round(1.5 * (max(0.01, kinetic_energy_megatons) ** 0.4), 2)
        yield_cbrt = max(0.001, kinetic_energy_megatons) ** (1.0 / 3.0)
        blast_20psi = round(yield_cbrt * 3.5, 2)
        blast_5psi = round(yield_cbrt * 8.2, 2)
        blast_1psi = round(yield_cbrt * 24.0, 2)

        return ImpactResults(
            is_atmospheric_skip=False,
            skip_warning_message=None,
            diameter_m=diameter_m,
            velocity_km_s=velocity_km_s,
            angle_deg=angle_deg,
            density_asteroid_kg_m3=density_i,
            density_target_kg_m3=density_t,
            mass_kg=mass_kg,
            mass_metric_tons=mass_metric_tons,
            kinetic_energy_joules=kinetic_energy_joules,
            kinetic_energy_megatons=kinetic_energy_megatons,
            hiroshima_bombs_equivalent=hiroshima_equiv,
            tunguska_equivalent=tunguska_equiv,
            chicxulub_equivalent=chicxulub_equiv,
            transient_crater_diameter_m=d_tc_m,
            transient_crater_diameter_km=d_tc_km,
            final_crater_diameter_m=d_final_m,
            final_crater_diameter_km=d_final_km,
            crater_depth_m=crater_depth_m,
            classification=classification,
            expected_environmental_effect=effect,
            severity_level=severity_level,
            theme_color=theme_color,
            richter_magnitude=richter_mag,
            fireball_radius_km=fireball_radius_km,
            blast_radius_20psi_heavy_km=blast_20psi,
            blast_radius_5psi_moderate_km=blast_5psi,
            blast_radius_1psi_light_km=blast_1psi,
            airburst_altitude_km=airburst_alt,
        )


def calculate_impact(
    diameter_m: float,
    velocity_km_s: float,
    angle_deg: float = 45.0,
    density_asteroid: float = DEFAULT_ASTEROID_DENSITY_RHO_I,
    density_target: float = DEFAULT_TARGET_DENSITY_RHO_T,
    explicit_mass_kg: Optional[float] = None,
) -> ImpactResults:
    """Convenience wrapper to run impact simulation."""
    params = ImpactParameters(
        diameter_m=diameter_m,
        velocity_m_s=velocity_km_s * 1000.0,
        angle_deg=angle_deg,
        density_asteroid_kg_m3=density_asteroid,
        density_target_kg_m3=density_target,
        mass_kg=explicit_mass_kg,
    )
    return ImpactPhysicsEngine.simulate(params)


# -----------------------------------------------------------------------------
# 3D PLOTLY VISUALIZERS & RIPPLE GENERATORS
# -----------------------------------------------------------------------------

def lat_lon_to_cartesian(lat_deg: float, lon_deg: float, radius: float = 1.0) -> Tuple[float, float, float]:
    """Convert spherical latitude/longitude to 3D Cartesian coordinates."""
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = radius * math.cos(lat_rad) * math.sin(lon_rad)
    z = radius * math.sin(lat_rad)
    return x, y, z


def generate_sphere_wireframe(radius: float = 1.0, num_steps: int = 40):
    """Generate 3D mesh surface points for the Earth globe."""
    u = np.linspace(0, 2 * np.pi, num_steps)
    v = np.linspace(0, np.pi, num_steps)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def generate_surface_circle(
    center_lat: float, center_lon: float, angular_radius_deg: float, radius: float = 1.002, num_pts: int = 60
) -> Tuple[List[float], List[float], List[float]]:
    """
    Generate 3D concentric ripple ring conforming to spherical Earth surface.
    """
    lat0 = math.radians(center_lat)
    lon0 = math.radians(center_lon)
    d = math.radians(angular_radius_deg)

    cx, cy, cz = [], [], []
    for bearing in np.linspace(0, 2 * np.pi, num_pts):
        lat = math.asin(math.sin(lat0) * math.cos(d) + math.cos(lat0) * math.sin(d) * math.cos(bearing))
        lon = lon0 + math.atan2(
            math.sin(bearing) * math.sin(d) * math.cos(lat0),
            math.cos(d) - math.sin(lat0) * math.sin(lat)
        )
        x = radius * math.cos(lat) * math.cos(lon)
        y = radius * math.cos(lat) * math.sin(lon)
        z = radius * math.sin(lat)
        cx.append(x)
        cy.append(y)
        cz.append(z)
    return cx, cy, cz


def build_3d_playground_canvas(
    results: ImpactResults,
    target_lat: float = 14.5,
    target_lon: float = -135.0,
    target_name: str = "Hypothetical Target",
) -> go.Figure:
    """
    Builds the interactive 3D Orbital Playground simulation canvas mockup:
    - 3D Earth Globe with continents grid and glowing atmospheric horizon
    - Dashed glowing red hypothetical trajectory line intersecting Earth
    - Glowing 'Impact Zone' radius with radar-like circular ripples
    - In case of skip (theta < 10 deg), shows upper-atmosphere rebound trajectory in cyan
    """
    fig = go.Figure()

    # 1. Earth Globe Wireframe Mesh
    xe, ye, ze = generate_sphere_wireframe(radius=1.0, num_steps=32)
    fig.add_trace(
        go.Surface(
            x=xe, y=ye, z=ze,
            colorscale=[
                [0.0, "#05162a"],
                [0.4, "#0b2a4a"],
                [0.7, "#083344"],
                [1.0, "#0e4a5e"],
            ],
            showscale=False,
            opacity=0.88,
            hoverinfo="none",
            name="Earth Globe",
        )
    )

    # 2. Atmospheric Halo Shell
    xa, ya, za = generate_sphere_wireframe(radius=1.06, num_steps=24)
    fig.add_trace(
        go.Surface(
            x=xa, y=ya, z=za,
            colorscale=[[0.0, "rgba(56, 189, 248, 0.05)"], [1.0, "rgba(168, 85, 247, 0.08)"]],
            showscale=False,
            opacity=0.15,
            hoverinfo="none",
            name="Atmosphere (100km)",
        )
    )

    # 3. Target Impact Coordinates on Earth
    tx, ty, tz = lat_lon_to_cartesian(target_lat, target_lon, radius=1.005)

    if not results.is_atmospheric_skip:
        # Trajectory Vector calculation
        theta_rad = math.radians(results.angle_deg)
        beam_dist = 2.4
        zenith_x, zenith_y, zenith_z = tx / 1.005, ty / 1.005, tz / 1.005
        traj_start_x = zenith_x * beam_dist * math.sin(theta_rad) + 1.2 * math.cos(theta_rad)
        traj_start_y = zenith_y * beam_dist * math.sin(theta_rad) + 0.8 * math.cos(theta_rad)
        traj_start_z = zenith_z * beam_dist * math.sin(theta_rad) + 1.0

        start_mag = math.sqrt(traj_start_x**2 + traj_start_y**2 + traj_start_z**2)
        sx = (traj_start_x / start_mag) * beam_dist
        sy = (traj_start_y / start_mag) * beam_dist
        sz = (traj_start_z / start_mag) * beam_dist

        # Inbound Trajectory Beam (Glowing Red Dashed Line)
        num_traj_pts = 20
        tx_line = np.linspace(sx, tx, num_traj_pts)
        ty_line = np.linspace(sy, ty, num_traj_pts)
        tz_line = np.linspace(sz, tz, num_traj_pts)

        fig.add_trace(
            go.Scatter3d(
                x=tx_line, y=ty_line, z=tz_line,
                mode="lines",
                line=dict(color="#ef4444", width=5, dash="dash"),
                name="Hypothetical Trajectory Vector",
                hoverinfo="text",
                text=[f"Inbound Asteroid Beam | Speed: {results.velocity_km_s:.1f} km/s"] * num_traj_pts,
            )
        )

        # Inbound Asteroid Impactor Marker (Start Point)
        fig.add_trace(
            go.Scatter3d(
                x=[sx], y=[sy], z=[sz],
                mode="markers+text",
                marker=dict(size=8, color="#a855f7", symbol="diamond"),
                text=["☄️ Impactor"],
                textposition="top center",
                name="Incoming Impactor",
                textfont=dict(family="JetBrains Mono", size=11, color="#c084fc"),
            )
        )

        # Impact Epicenter Marker (Red pulsing node)
        fig.add_trace(
            go.Scatter3d(
                x=[tx], y=[ty], z=[tz],
                mode="markers+text",
                marker=dict(size=10, color="#f43f5e", symbol="circle"),
                text=[f"💥 Impact Epicenter\n({target_lat:.1f}°, {target_lon:.1f}°)"],
                textposition="bottom center",
                name="Impact Zone",
                textfont=dict(family="JetBrains Mono", size=11, color="#fda4af"),
            )
        )

        # Concentric Radar Ripples on Earth's Surface
        # Ring 1: Transient Crater Zone (Pi-Scaling)
        crater_ang_deg = max(0.5, min(15.0, (results.transient_crater_diameter_km / EARTH_RADIUS_KM) * (180.0 / math.pi) * 8.0))
        r1_x, r1_y, r1_z = generate_surface_circle(target_lat, target_lon, crater_ang_deg, radius=1.008)
        fig.add_trace(
            go.Scatter3d(
                x=r1_x, y=r1_y, z=r1_z,
                mode="lines",
                line=dict(color="#f43f5e", width=4),
                name=f"Crater Rim ({results.transient_crater_diameter_km:.1f} km)",
                hoverinfo="name",
            )
        )

        # Ring 2: 20 psi Extreme Overpressure Shockwave Zone
        blast_20_ang = max(1.2, min(25.0, (results.blast_radius_20psi_heavy_km / EARTH_RADIUS_KM) * (180.0 / math.pi)))
        r2_x, r2_y, r2_z = generate_surface_circle(target_lat, target_lon, blast_20_ang, radius=1.010)
        fig.add_trace(
            go.Scatter3d(
                x=r2_x, y=r2_y, z=r2_z,
                mode="lines",
                line=dict(color="#f97316", width=3, dash="dot"),
                name=f"20 psi Blast Shockwave ({results.blast_radius_20psi_heavy_km:.1f} km)",
                hoverinfo="name",
            )
        )

        # Ring 3: 5 psi Severe Damage Zone (Radar Ripple)
        blast_5_ang = max(2.0, min(40.0, (results.blast_radius_5psi_moderate_km / EARTH_RADIUS_KM) * (180.0 / math.pi)))
        r3_x, r3_y, r3_z = generate_surface_circle(target_lat, target_lon, blast_5_ang, radius=1.012)
        fig.add_trace(
            go.Scatter3d(
                x=r3_x, y=r3_y, z=r3_z,
                mode="lines",
                line=dict(color="#fbbf24", width=2, dash="dash"),
                name=f"5 psi Moderate Collapse Zone ({results.blast_radius_5psi_moderate_km:.1f} km)",
                hoverinfo="name",
            )
        )

        # Ring 4: Thermal Radiation / Fireball Horizon
        fireball_ang = max(3.0, min(55.0, (results.fireball_radius_km / EARTH_RADIUS_KM) * (180.0 / math.pi)))
        r4_x, r4_y, r4_z = generate_surface_circle(target_lat, target_lon, fireball_ang, radius=1.014)
        fig.add_trace(
            go.Scatter3d(
                x=r4_x, y=r4_y, z=r4_z,
                mode="lines",
                line=dict(color="#a855f7", width=2, dash="longdash"),
                name=f"Thermal Fireball Horizon ({results.fireball_radius_km:.1f} km)",
                hoverinfo="name",
            )
        )

    else:
        # ATMOSPHERIC SKIP TRAJECTORY (theta < 10 degrees)
        t_vals = np.linspace(-1.8, 1.8, 40)
        tangent_x, tangent_y, tangent_z = -ty, tx, 0.2
        t_mag = math.sqrt(tangent_x**2 + tangent_y**2 + tangent_z**2)
        tangent_x, tangent_y, tangent_z = tangent_x/t_mag, tangent_y/t_mag, tangent_z/t_mag

        zenith_x, zenith_y, zenith_z = tx / 1.005, ty / 1.005, tz / 1.005

        arc_x = [tx * 1.02 + tangent_x * t + zenith_x * (0.35 * (t**2)) for t in t_vals]
        arc_y = [ty * 1.02 + tangent_y * t + zenith_y * (0.35 * (t**2)) for t in t_vals]
        arc_z = [tz * 1.02 + tangent_z * t + zenith_z * (0.35 * (t**2)) for t in t_vals]

        fig.add_trace(
            go.Scatter3d(
                x=arc_x, y=arc_y, z=arc_z,
                mode="lines",
                line=dict(color="#38bdf8", width=6),
                name="Deflected Upper-Atmosphere Arc (Skip)",
                hoverinfo="name",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[tx * 1.02], y=[ty * 1.02], z=[tz * 1.02],
                mode="markers+text",
                marker=dict(size=9, color="#38bdf8", symbol="cross"),
                text=["↩️ Atmospheric Perigee & Ricochet (~85km)"],
                textposition="top center",
                name="Skip Perigee",
                textfont=dict(family="JetBrains Mono", size=11, color="#7dd3fc"),
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            bgcolor="#030712",
            camera=dict(
                eye=dict(x=1.5, y=1.2, z=1.1),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        paper_bgcolor="#030712",
        plot_bgcolor="#030712",
        height=580,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
            font=dict(family="JetBrains Mono", size=11, color="#94a3b8"),
        ),
    )
    return fig


def build_energy_comparison_chart(results: ImpactResults) -> go.Figure:
    """Builds horizontal comparative yield bar chart against famous events."""
    benchmarks = [
        ("Hiroshima A-Bomb (1945)", 0.015, "#10b981"),
        ("Chelyabinsk Airburst (2013)", 0.50, "#f59e0b"),
        ("Tunguska Taiga Event (1908)", 15.0, "#f97316"),
        ("Tsar Bomba Detonation (1961)", 50.0, "#ef4444"),
        ("Current Simulation Impactor", max(0.001, results.kinetic_energy_megatons), "#c084fc"),
        ("Chicxulub Dinosaur Killer", 100_000_000.0, "#a855f7"),
    ]

    labels = [b[0] for b in benchmarks]
    values = [b[1] for b in benchmarks]
    colors = [b[2] for b in benchmarks]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#1e293b", width=1)),
            text=[f"{v:,.2f} Mt" if v < 1000 else f"{v:,.0f} Mt" for v in values],
            textposition="auto",
            textfont=dict(family="JetBrains Mono", size=11, color="#f8fafc"),
        )
    )

    fig.update_layout(
        xaxis_type="log",
        xaxis=dict(
            title="Kinetic Yield (Megatons of TNT - Log Scale)",
            gridcolor="#1e293b",
            color="#94a3b8",
            tickfont=dict(family="JetBrains Mono", size=10),
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            color="#cbd5e1",
            tickfont=dict(family="Inter", size=11, weight="bold"),
        ),
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        height=260,
        margin=dict(l=10, r=20, t=20, b=30),
    )
    return fig
