"""Simulation package for FL scenario simulation.

Provides:
- Client simulators (honest and Byzantine)
- Metrics collection
- Complete FL simulation runner
"""

from src.simulation.clients import (
    HonestClientSimulator,
    ByzantineClientSimulator,
    create_client_simulators
)
from src.simulation.metrics import MetricsCollector
from src.simulation.runner import FLSimulation, SimulationConfig, run_simulation

__all__ = [
    'HonestClientSimulator',
    'ByzantineClientSimulator',
    'create_client_simulators',
    'MetricsCollector',
    'FLSimulation',
    'SimulationConfig',
    'run_simulation'
]
