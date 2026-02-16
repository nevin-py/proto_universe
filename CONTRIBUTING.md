# Contributing to ProtoGalaxy

## Development Setup

### 1. Clone and Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Vanesor/proto_universe.git
cd proto_universe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Build ZKP Module

```bash
cd sonobe/fl-zkp-bridge
maturin develop --release
cd ../..
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_bounded_zkp.py -xvs

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Development Workflow

### Making Changes to Python Code

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test:
   ```bash
   # Run tests
   pytest tests/

   # Format code
   black src/ scripts/ tests/
   isort src/ scripts/ tests/

   # Type checking
   mypy src/
   ```

3. Commit and push:
   ```bash
   git add .
   git commit -m "feat: your feature description"
   git push origin feature/your-feature-name
   ```

### Making Changes to Sonobe (ZKP Module)

The `sonobe/` directory is a git submodule pointing to your fork at `Vanesor/sonobe`.

1. Make changes in `sonobe/fl-zkp-bridge/src/lib.rs`

2. Test locally:
   ```bash
   cd sonobe/fl-zkp-bridge
   cargo test
   maturin develop --release
   cd ../..
   python -c "from fl_zkp_bridge import FLZKPBoundedProver; print('OK')"
   ```

3. Commit to sonobe:
   ```bash
   cd sonobe
   git add fl-zkp-bridge/src/lib.rs
   git commit -m "feat: your ZKP change"
   git push origin main
   cd ..
   ```

4. Update proto_universe to use new sonobe commit:
   ```bash
   git add sonobe
   git commit -m "chore: update sonobe submodule"
   git push origin main
   ```

### Updating Dependencies

When adding new Python dependencies:

1. Add to `requirements.txt` (production) or `requirements-dev.txt` (development)
2. Update `setup.py` if it's a core dependency
3. Test clean installation:
   ```bash
   pip install -r requirements.txt
   ```

## Code Style

- **Python**: Follow PEP 8, use Black for formatting (line length 100)
- **Rust**: Follow Rust style guide, use `cargo fmt`
- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Use Google-style docstrings

## Testing Guidelines

- Write tests for all new features
- Maintain >80% code coverage
- Use descriptive test names: `test_<feature>_<scenario>_<expected_result>`
- For ZKP tests: use deterministic seeds and small scale factors

## Commit Messages

Follow conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding/updating tests
- `refactor:` - Code change that neither fixes a bug nor adds a feature
- `perf:` - Performance improvement
- `chore:` - Maintenance tasks

## Pull Request Process

1. Create PR against `main` branch
2. Ensure all tests pass
3. Update documentation if needed
4. Request review from maintainers
5. Address review comments
6. Squash commits if requested

## Questions?

Open an issue or reach out to the team!
