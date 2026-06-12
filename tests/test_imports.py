"""The package imports cleanly and every advertised name exists."""
import rivabar as rb


def test_import_and_all_resolve():
    missing = [name for name in rb.__all__ if not hasattr(rb, name)]
    assert not missing, f"__all__ names missing from package: {missing}"


def test_extract_centerline_is_alias():
    # Backward-compatible alias for the functional API entry point
    assert rb.extract_centerline.__module__ == 'rivabar.core'
