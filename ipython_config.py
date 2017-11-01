c = get_config()

c.InteractiveShellApp.exec_lines = [
    '%load_ext autoreload',
    '%autoreload 2',
    'from datetime import datetime, timedelta',
    'from analyst_class import Analyst',
    'import utils',
    'import analyst_functions as af',
    'import trader'
]
