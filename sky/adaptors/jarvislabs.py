"""JarvisLabs cloud adaptor."""
from sky.adaptors import common

jarvislabs = common.LazyImport(
    'jarvislabs',
    import_error_message='Failed to import dependencies for JarvisLabs. '
    'Try running: pip install "skypilot[jarvislabs]"')
