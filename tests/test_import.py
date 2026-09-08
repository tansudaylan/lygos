import importlib


def test_import_lygos_package():
    lygos = importlib.import_module('lygos')
    assert hasattr(lygos, '__file__')


def test_import_lygos_main_module():
    main = importlib.import_module('lygos.main')
    assert hasattr(main, 'init')
