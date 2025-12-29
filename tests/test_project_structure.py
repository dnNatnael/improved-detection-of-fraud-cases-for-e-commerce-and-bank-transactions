"""
Basic CI-safe tests for fraud-detection project
These tests:
- Do NOT load datasets
- Do NOT train models
- Do NOT run notebooks
- Only verify structure and imports
"""

def test_src_package_exists():
    import src
    assert src is not None


def test_notebooks_package_exists():
    import notebooks
    assert notebooks is not None


def test_scripts_package_exists():
    import scripts
    assert scripts is not None


def test_requirements_file_exists():
    import os
    assert os.path.exists("requirements.txt")


def test_data_directory_not_required():
    """
    Data directory should NOT be required for CI to pass.
    """
    import os
    assert not os.path.exists("data") or os.path.isdir("data")
