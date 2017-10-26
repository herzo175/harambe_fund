c = get_config()

c.InteractiveShellApp.exec_lines = [
		'%load_ext autoreload',
		'%autoreload 2',
		'from datetime import datetime, timedelta',
    'from analyst import Analyst',
    'import scrapers'
]
