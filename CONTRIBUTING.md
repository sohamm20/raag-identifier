# Contributing to Raag Identifier

Thank you for your interest in contributing to Raag Identifier! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/raag-identifier.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Run tests to ensure everything works: `pytest tests/`

## Development Workflow

1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Run tests: `pytest tests/ -v`
5. Ensure code quality:
   - Follow PEP 8 style guidelines
   - Add docstrings to new functions/classes
   - Add type hints where appropriate
6. Commit your changes: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Code Style

- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add docstrings to all functions, classes, and modules
- Use type hints for function parameters and return values
- Keep functions focused and concise
- Comment complex logic

Example:
```python
def extract_features(
    audio: np.ndarray,
    sample_rate: int = 22050,
    use_cqt: bool = True,
) -> np.ndarray:
    """
    Extract time-frequency features from audio.

    Args:
        audio: Input audio signal
        sample_rate: Audio sample rate in Hz
        use_cqt: Whether to use CQT (True) or Mel spectrogram (False)

    Returns:
        Feature array with shape [n_features, time_frames]
    """
    # Implementation here
    pass
```

## Testing

- Write unit tests for all new features
- Tests should be in the `tests/` directory
- Use pytest for testing
- Aim for >80% code coverage
- Test edge cases and error conditions

Run tests:
```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src/raag_identifier --cov-report=html

# Specific test file
pytest tests/test_models.py -v
```

## Documentation

- Update README.md if you add new features or change usage
- Add docstrings to all public functions and classes
- Include examples in docstrings when helpful
- Update configuration documentation if you add new parameters

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add a clear description of your changes
4. Reference any related issues
5. Wait for code review
6. Address review feedback
7. Once approved, your PR will be merged

## Types of Contributions

### Bug Fixes
- Check if the bug is already reported in Issues
- Create a new issue if not
- Reference the issue in your PR

### New Features
- Discuss major features in an issue first
- Ensure features align with project goals
- Add comprehensive tests
- Update documentation

### Performance Improvements
- Include benchmarks showing improvement
- Ensure no accuracy regression
- Add performance tests if applicable

### Documentation
- Fix typos, clarify explanations
- Add examples and tutorials
- Improve code comments

## Project Structure

```
Raag/
├── src/raag_identifier/    # Main package
│   ├── data/              # Dataset loaders
│   ├── models/            # Model architectures
│   ├── preprocessing/     # Audio preprocessing
│   └── utils/             # Utilities
├── tests/                 # Unit tests
├── config/               # Configuration files
├── train.py             # Training script
├── inference.py         # Inference script
└── evaluate.py          # Evaluation script
```

## Code Review Checklist

Before submitting a PR, ensure:
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] No unnecessary dependencies added
- [ ] Commits are clear and descriptive
- [ ] No debugging code left in

## Questions?

If you have questions:
- Check existing issues and discussions
- Open a new issue for discussion
- Be specific and provide context

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in the project README. Thank you for helping improve Raag Identifier!
