# Sonobe ProtoGalaxy Python Bindings

Python bindings for the ProtoGalaxy folding scheme with variable k (multi-instance folding) support.

## Features

- Support for variable k (number of incoming instances)
- Power-of-two constraint validation (k+1 must be a power of 2)
- Simplified Python interface to ProtoGalaxy
- Type-safe parameter validation

## Installation

### Build from source

```bash
cd sonobe-py-bindings
maturin develop  # For development
# or
maturin build --release  # For production wheel
```

### Install the wheel

```bash
pip install target/wheels/sonobe_protogalaxy-*.whl
```

## Usage

```python
import sonobe_protogalaxy

# Create ProtoGalaxy instance with k=1 (single-instance folding)
pg = sonobe_protogalaxy.PyProtoGalaxy(k=1)
print(pg)  # PyProtoGalaxy(k=1, total_instances=2)

# Create ProtoGalaxy instance with k=3 (multi-instance folding)
pg = sonobe_protogalaxy.PyProtoGalaxy(k=3)
print(pg.total_instances())  # 4

# Invalid k values will raise ValueError
try:
    pg = sonobe_protogalaxy.PyProtoGalaxy(k=2)  # k+1=3 is not a power of 2
except ValueError as e:
    print(f"Error: {e}")
```

## Valid k Values

k must satisfy two constraints:
1. k >= 1
2. k + 1 must be a power of two

Valid k values: 1, 3, 7, 15, 31, 63, 127, ...

## Testing

```bash
# Run tests with pytest
pytest tests/test_variable_k.py -v

# Or run directly
python tests/test_variable_k.py
```

## Requirements

- Python >= 3.8
- Rust toolchain (for building)
- maturin >= 1.0

## License

Same as sonobe library
