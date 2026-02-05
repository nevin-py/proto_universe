from setuptools import setup, find_packages

setup(
    name="proto_system",
    version="0.1.0",
    description="Privacy-preserving federated learning with Byzantine-resilient aggregation",
    author="ProtoGalaxy Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
    ],
)
