from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="proto_universe",
    version="0.1.0",
    description="FiZK: Privacy-preserving federated learning with 5-layer Byzantine-resilient defense and ZKP verification",
    author="FiZK Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "zkp": [
            # Install ZKP Rust bindings separately:
            # cd sonobe/fl-zkp-bridge && maturin develop --release
        ],
    },
    entry_points={
        "console_scripts": [
            "proto-eval=scripts.run_evaluation:main",
        ],
    },
)
