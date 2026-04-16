"""
Seed PostgreSQL with 50 historical anomaly records spanning all 3 anomaly
types and all 5 sensor IDs.

Schema created (if not exists):
    Table: historical_anomalies
    Columns: id (UUID PK), sensor_id, anomaly_type, detected_at,
             duration_minutes, severity, root_cause, resolution,
             resolved_by, resolution_time_hours

Run:
    python scripts/seed_postgres.py

Requires:
    POSTGRES_URL environment variable (set in .env)
"""

from __future__ import annotations

import os
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Session

load_dotenv(Path(__file__).parent.parent / ".env")

# ── DB connection ─────────────────────────────────────────────────────────────

POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL", "postgresql://postgres:password@localhost:5432/anomaly_db"
)

# ── ORM model ─────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class HistoricalAnomaly(Base):
    """ORM model for the historical_anomalies table."""

    __tablename__ = "historical_anomalies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sensor_id = Column(String(20), nullable=False, index=True)
    anomaly_type = Column(String(50), nullable=False, index=True)
    detected_at = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    severity = Column(Float, nullable=False)
    root_cause = Column(Text, nullable=False)
    resolution = Column(Text, nullable=False)
    resolved_by = Column(String(100), nullable=False)
    resolution_time_hours = Column(Float, nullable=False)


# ── Seed data ─────────────────────────────────────────────────────────────────

SENSOR_IDS = ["PUMP_01", "PUMP_02", "PUMP_03", "MOTOR_01", "MOTOR_02"]

BEARING_WEAR_RECORDS = [
    {
        "root_cause": (
            "Lubrication failure due to degraded grease in the main drive bearing. "
            "Vibration amplitude increased from 0.25 mm/s to 3.8 mm/s over 72 hours. "
            "Oil sample analysis confirmed metal particle contamination indicating "
            "early-stage spalling on the inner race."
        ),
        "resolution": (
            "Emergency bearing replacement with SKF 6311-2RS1 deep groove ball bearing. "
            "Grease nipple cleaned and repacked with Mobil Mobilith SHC 460. "
            "Vibration returned to baseline 0.22 mm/s within 2 hours of restart."
        ),
        "resolved_by": "Technician Klaus Bauer",
        "duration_minutes": 210,
        "resolution_time_hours": 4.5,
        "severity": 0.72,
    },
    {
        "root_cause": (
            "Bearing inner race fatigue caused by prolonged operation above rated RPM. "
            "Temperature sensors recorded sustained 98°C over 48 hours, exceeding the "
            "95°C alarm threshold. Acoustic emission analysis detected characteristic "
            "bearing spall frequency at 147 Hz."
        ),
        "resolution": (
            "Replaced both drive-end and non-drive-end bearings. Installed PT100 "
            "temperature sensor directly on bearing housing for improved monitoring. "
            "RPM setpoint reduced from 1600 to 1520 to prevent recurrence."
        ),
        "resolved_by": "Engineer Sarah Mitchell",
        "duration_minutes": 340,
        "resolution_time_hours": 6.0,
        "severity": 0.81,
    },
    {
        "root_cause": (
            "Misalignment between pump shaft and motor shaft caused uneven load on "
            "coupling-side bearing. Laser alignment check revealed 0.45 mm angular "
            "misalignment. Vibration signature showed 2x running speed harmonic "
            "characteristic of misalignment-induced bearing stress."
        ),
        "resolution": (
            "Shaft realigned using Pruftechnik OPTALIGN laser system to within 0.02 mm. "
            "Coupling-side bearing replaced preventively. Alignment verification carried "
            "out after thermal stabilisation at operating temperature."
        ),
        "resolved_by": "Technician Jan Kowalski",
        "duration_minutes": 180,
        "resolution_time_hours": 5.0,
        "severity": 0.63,
    },
    {
        "root_cause": (
            "Foreign particle ingress through damaged bearing seal. Inspection revealed "
            "a torn lip seal allowing process fluid contamination. Contaminated lubricant "
            "caused abrasive wear on rolling elements, confirmed by 40% increase in "
            "vibration RMS over 5 days."
        ),
        "resolution": (
            "Bearing replaced and lip seal upgraded to a V-ring seal with improved "
            "chemical resistance. Lubricant flushed and replaced. New seal material "
            "rated for process fluid pH 3.5. Post-repair vibration: 0.19 mm/s."
        ),
        "resolved_by": "Maintenance Lead Ahmed Farouk",
        "duration_minutes": 260,
        "resolution_time_hours": 5.5,
        "severity": 0.68,
    },
    {
        "root_cause": (
            "Bearing overloading due to blocked discharge line creating excessive radial "
            "load. Back-pressure caused shaft deflection, leading to edge loading on "
            "bearing outer race. Vibration reached 4.2 mm/s before automatic shutdown."
        ),
        "resolution": (
            "Discharge blockage cleared — found solidified process residue at check "
            "valve. Bearing replaced. Discharge pressure monitoring added to SCADA "
            "alarm system with 6.5 bar high-high trip set point."
        ),
        "resolved_by": "Operator Maria Gonzalez",
        "duration_minutes": 95,
        "resolution_time_hours": 2.0,
        "severity": 0.77,
    },
]

PRESSURE_DROP_RECORDS = [
    {
        "root_cause": (
            "Partial blockage of pump inlet strainer reduced suction pressure below "
            "the minimum NPSH required, inducing cavitation. Pressure dropped from "
            "5.1 bar to 2.3 bar within 15 minutes. Acoustic signs of cavitation heard "
            "at pump casing."
        ),
        "resolution": (
            "Inlet strainer cleaned and debris removed — found process scale buildup "
            "reducing effective flow area by 65%. Strainer differential pressure alarm "
            "set at 0.3 bar. Cavitation protection interlock activated."
        ),
        "resolved_by": "Operator Thomas Richter",
        "duration_minutes": 45,
        "resolution_time_hours": 1.5,
        "severity": 0.59,
    },
    {
        "root_cause": (
            "Rupture of 25 mm bypass line gasket downstream of pump discharge caused "
            "sudden pressure loss. Process fluid leak rate estimated at 12 L/min. "
            "Pressure fell from 5.0 bar to 1.8 bar in under 3 minutes triggering the "
            "low-pressure trip."
        ),
        "resolution": (
            "Emergency isolation of bypass line. Spiral-wound graphite gasket replaced "
            "with PTFE-encapsulated type rated for 16 bar and process fluid compatibility. "
            "Flanged joint torqued to 85 Nm per procedure. Leak-tested at 7.5 bar."
        ),
        "resolved_by": "Pipefitter Leon Dubois",
        "duration_minutes": 120,
        "resolution_time_hours": 3.0,
        "severity": 0.74,
    },
    {
        "root_cause": (
            "Control valve CV-201 failed open due to instrument air solenoid fault. "
            "Valve moved from 40% to fully open, causing an uncontrolled pressure "
            "relief and drop from 4.9 bar to 2.1 bar. Solenoid coil resistance "
            "measured at open circuit."
        ),
        "resolution": (
            "Solenoid coil replaced on CV-201 actuator. Valve stroke-tested and "
            "calibrated to 4–20 mA signal. Instrument air supply filter cleaned. "
            "Redundant pressure transmitter PT-202 cross-check alarm enabled in DCS."
        ),
        "resolved_by": "Instrument Technician Yuki Tanaka",
        "duration_minutes": 75,
        "resolution_time_hours": 2.5,
        "severity": 0.65,
    },
    {
        "root_cause": (
            "Worn pump impeller reduced hydraulic efficiency by 22%, insufficient to "
            "maintain system pressure against rising discharge head. Wear ring clearance "
            "measured at 1.8 mm against the acceptable maximum of 0.5 mm, causing "
            "significant internal recirculation."
        ),
        "resolution": (
            "Impeller replaced with OEM part (part no. IMP-4450-SS316). Wear rings "
            "replaced and clearances set to 0.15 mm. Pump performance curve retested — "
            "achieved 98% of design head at rated flow. Impeller inspection added to "
            "annual PM schedule."
        ),
        "resolved_by": "Engineer Paulo Ferreira",
        "duration_minutes": 480,
        "resolution_time_hours": 10.0,
        "severity": 0.56,
    },
    {
        "root_cause": (
            "Air entrainment in suction line following partial open of vent valve "
            "during operator error. Air pockets caused pressure oscillations and net "
            "pressure dropped to 2.8 bar. Dissolved oxygen sensor confirmed air "
            "ingress at 8.4 ppm."
        ),
        "resolution": (
            "Vent valve fully closed and locked out. System primed to purge entrained "
            "air. Operational procedure updated requiring two-person sign-off for "
            "suction line vent operations. Pressure stabilised at 4.95 bar."
        ),
        "resolved_by": "Shift Supervisor Ingrid Larsson",
        "duration_minutes": 35,
        "resolution_time_hours": 1.0,
        "severity": 0.48,
    },
]

OVERLOAD_RECORDS = [
    {
        "root_cause": (
            "Seized pump shaft caused motor to draw 38 A against rated 14 A. "
            "Root cause found to be a broken key in the shaft keyway, allowing the "
            "shaft to rotate inside the impeller hub and jam. Current spike tripped "
            "the motor protection relay after 8 seconds."
        ),
        "resolution": (
            "Shaft assembly disassembled and broken key removed. Keyway inspected for "
            "fretting damage — 0.3 mm wear found and shaft replaced. New stainless "
            "steel key fitted and secured with Loctite 243. Motor restarted and "
            "current verified at 13.2 A."
        ),
        "resolved_by": "Mechanical Engineer Carlos Reyes",
        "duration_minutes": 420,
        "resolution_time_hours": 8.0,
        "severity": 0.88,
    },
    {
        "root_cause": (
            "Incorrect viscosity process fluid (80 cP instead of rated 15 cP) routed "
            "to pump following tank mix-up. High fluid resistance caused motor current "
            "to rise steadily from 13 A to 31 A over 20 minutes. RPM dropped 18% due "
            "to slip increase."
        ),
        "resolution": (
            "Pump isolated and correct fluid batch rerouted. Motor allowed to cool for "
            "45 minutes before restart. Winding insulation resistance tested at 850 MΩ "
            "— acceptable. Tank labelling procedure revised with barcode verification. "
            "Current normalised at 13.6 A."
        ),
        "resolved_by": "Process Engineer Lisa Chen",
        "duration_minutes": 65,
        "resolution_time_hours": 2.0,
        "severity": 0.71,
    },
    {
        "root_cause": (
            "Phase imbalance of 8.5% on the motor supply caused unequal current "
            "distribution, raising total draw to 29 A. Thermal imaging of the MCC "
            "panel revealed loose connection on L2 phase bus bar, increasing contact "
            "resistance to 0.42 Ω."
        ),
        "resolution": (
            "MCC isolated under permit-to-work. L2 bus bar connection tightened to "
            "specified 40 Nm torque. Contact surfaces cleaned and re-torqued. Phase "
            "imbalance reduced to 0.8%. Motor current balanced at 13.1 A across all "
            "three phases."
        ),
        "resolved_by": "Electrician Andrei Popescu",
        "duration_minutes": 90,
        "resolution_time_hours": 3.5,
        "severity": 0.67,
    },
    {
        "root_cause": (
            "Blocked discharge caused dead-head operation. Motor ran against closed "
            "isolation valve for 12 minutes, drawing 36 A with RPM falling to 1180. "
            "Motor winding temperature reached 148°C measured by embedded PT100. "
            "Automatic trip did not activate — found that thermal relay setting had "
            "been incorrectly adjusted to 45 A during last calibration."
        ),
        "resolution": (
            "Discharge valve opened, motor cooled, and winding insulation tested. "
            "Thermal relay reset to correct 16 A trip setting per motor nameplate. "
            "Blocked valve root cause: accidental closure by contractor during "
            "adjacent pipework maintenance. Permit-to-work isolation boundary extended."
        ),
        "resolved_by": "Maintenance Supervisor Derek Walsh",
        "duration_minutes": 185,
        "resolution_time_hours": 4.0,
        "severity": 0.85,
    },
    {
        "root_cause": (
            "VFD fault caused motor to accelerate beyond synchronous speed, drawing "
            "33 A and causing RPM to reach 1820 before overspeed protection activated. "
            "VFD IGBT gate driver fault code F0023 recorded in event log. Capacitor "
            "bank voltage ripple measured at 12% above specification."
        ),
        "resolution": (
            "VFD DC bus capacitors replaced (service life: 7 years, found at 9 years). "
            "IGBT gate driver board replaced under manufacturer warranty. VFD "
            "recommissioned with ramp-up time extended from 5 s to 12 s to reduce "
            "inrush. Motor current verified at 13.4 A under full load."
        ),
        "resolved_by": "VFD Specialist Nadia Okonkwo",
        "duration_minutes": 300,
        "resolution_time_hours": 7.0,
        "severity": 0.79,
    },
]


def _build_records(
    rng: random.Random,
    base_dt: datetime,
) -> list[dict]:
    """Combine all record templates into 50 shuffled HistoricalAnomaly dicts."""
    records: list[dict] = []

    # Build one entry per template, cycling through sensors
    templates = [
        ("bearing_wear", BEARING_WEAR_RECORDS),
        ("pressure_drop", PRESSURE_DROP_RECORDS),
        ("overload", OVERLOAD_RECORDS),
    ]

    sensor_cycle = SENSOR_IDS * 20   # enough to pick from
    rng.shuffle(sensor_cycle)
    sensor_iter = iter(sensor_cycle)

    for anomaly_type, tmpl_list in templates:
        # Repeat templates to fill ~16-17 records per type (total = 50)
        repeat = 3 if anomaly_type == "bearing_wear" else 3
        for i, tmpl in enumerate(tmpl_list * repeat):
            if len([r for r in records if r["anomaly_type"] == anomaly_type]) >= 17:
                break
            offset_days = rng.randint(0, 180)
            offset_hours = rng.randint(0, 23)
            detected_at = base_dt - timedelta(days=offset_days, hours=offset_hours)
            records.append(
                {
                    "id": uuid.uuid4(),
                    "sensor_id": next(sensor_iter),
                    "anomaly_type": anomaly_type,
                    "detected_at": detected_at,
                    "duration_minutes": tmpl["duration_minutes"]
                    + rng.randint(-30, 30),
                    "severity": round(
                        min(1.0, max(0.1, tmpl["severity"] + rng.uniform(-0.05, 0.05))),
                        3,
                    ),
                    "root_cause": tmpl["root_cause"],
                    "resolution": tmpl["resolution"],
                    "resolved_by": tmpl["resolved_by"],
                    "resolution_time_hours": round(
                        tmpl["resolution_time_hours"] + rng.uniform(-0.5, 0.5), 1
                    ),
                }
            )

    # Pad to exactly 50 if needed
    while len(records) < 50:
        tmpl = rng.choice(BEARING_WEAR_RECORDS)
        offset_days = rng.randint(0, 90)
        records.append(
            {
                "id": uuid.uuid4(),
                "sensor_id": rng.choice(SENSOR_IDS),
                "anomaly_type": "bearing_wear",
                "detected_at": base_dt - timedelta(days=offset_days),
                "duration_minutes": tmpl["duration_minutes"],
                "severity": tmpl["severity"],
                "root_cause": tmpl["root_cause"],
                "resolution": tmpl["resolution"],
                "resolved_by": tmpl["resolved_by"],
                "resolution_time_hours": tmpl["resolution_time_hours"],
            }
        )

    rng.shuffle(records)
    return records[:50]


# ── Main ──────────────────────────────────────────────────────────────────────


def seed(postgres_url: str | None = None) -> None:
    """Create the schema and insert 50 historical anomaly records.

    Args:
        postgres_url: Override the POSTGRES_URL env var (used in tests).
    """
    url = postgres_url or POSTGRES_URL
    engine = create_engine(url, echo=False)

    print(f"Connecting to: {url}")
    Base.metadata.create_all(engine)
    print("Schema created / verified.")

    rng = random.Random(42)
    base_dt = datetime(2024, 6, 1, 12, 0, 0)
    records = _build_records(rng, base_dt)

    with Session(engine) as session:
        # Clear existing data to make seeding idempotent
        session.execute(text("DELETE FROM historical_anomalies"))
        session.bulk_insert_mappings(HistoricalAnomaly, records)  # type: ignore[arg-type]
        session.commit()

    print(f"\nInserted {len(records)} records.\n")
    _print_summary(engine)


def _print_summary(engine) -> None:
    """Print verification queries."""
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM historical_anomalies")
        ).scalar()
        print(f"Total records : {total}")

        print("\nBy anomaly_type:")
        rows = conn.execute(
            text(
                "SELECT anomaly_type, COUNT(*) AS n "
                "FROM historical_anomalies GROUP BY anomaly_type ORDER BY anomaly_type"
            )
        ).fetchall()
        for row in rows:
            print(f"  {row[0]:<20} {row[1]:>3}")

        print("\nBy sensor_id:")
        rows = conn.execute(
            text(
                "SELECT sensor_id, COUNT(*) AS n "
                "FROM historical_anomalies GROUP BY sensor_id ORDER BY sensor_id"
            )
        ).fetchall()
        for row in rows:
            print(f"  {row[0]:<12} {row[1]:>3}")


if __name__ == "__main__":
    seed()
