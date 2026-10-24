import importlib


def test_import_lygos_package():
    lygos = importlib.import_module('lygos')
    assert hasattr(lygos, '__file__')


def test_import_lygos_main_module():
    main = importlib.import_module('lygos.main')
    assert hasattr(main, 'init')


def test_pytest_configuration_collects_duplicate_module_names(tmp_path):
    from pathlib import Path
    import subprocess
    import sys

    files = []
    for name in ('first', 'second'):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / 'test_shared_name.py'
        path.write_text('def test_collected():\n    pass\n')
        files.append(str(path))
    config = Path(__file__).parents[1] / 'pyproject.toml'
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '--collect-only', '-c', str(config), *files],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '2 tests collected' in result.stdout
