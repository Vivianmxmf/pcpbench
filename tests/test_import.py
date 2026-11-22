def test_package_imports_and_has_version():
    import pcpbench
    assert isinstance(pcpbench.__version__, str)
    assert pcpbench.__version__
