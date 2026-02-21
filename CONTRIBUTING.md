# Contributing to StateBench

Thank you for your interest in contributing to StateBench! This project aims to provide a rigorous, fair benchmark for evaluating LLM memory systems.

## Ways to Contribute

### Add New Benchmark Tracks
We welcome new evaluation tracks that test distinct memory challenges. A good track should:
- Target a specific, well-defined memory failure mode
- Be synthetic and reproducible (no reliance on external data)
- Include clear ground truth for automated evaluation
- Be domain-agnostic or cover a new domain

### Add New Baselines
Help make the benchmark more comprehensive by adding memory strategies:
- Implement the `MemoryStrategy` interface in `src/statebench/baselines/`
- Include a brief description of the approach
- Ensure it operates within the standard token budget for fair comparison

### Improve Evaluation
- Enhance the LLM-as-judge prompts for better accuracy
- Add new metrics that capture important memory behaviors
- Improve deterministic checks for ground truth matching

### Expand Templates
- Add new templates to existing tracks for greater diversity
- Cover additional domains (healthcare, legal, education, etc.)
- Create adversarial variants that stress-test edge cases

### Documentation and Examples
- Improve documentation clarity
- Add usage examples and tutorials
- Share benchmark results with proper methodology

## Development Setup

```bash
# Clone the repository
git clone https://github.com/mliotta/statebench.git
cd statebench

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with clear, descriptive commits
3. Ensure tests pass and add new tests for new functionality
4. Update documentation as needed
5. Submit a pull request with a clear description of the changes

## Code Style

- Follow existing code patterns in the repository
- Use type hints for function signatures
- Keep functions focused and well-documented
- Run `ruff` before submitting

## Reporting Issues

When reporting bugs or suggesting features:
- Check existing issues first to avoid duplicates
- Provide clear reproduction steps for bugs
- For feature requests, explain the use case and expected behavior

## Questions?

Open an issue for questions about contributing. We're happy to help!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
