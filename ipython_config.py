c = get_config()

c.InteractiveShellApp.extensions = ['autoreload']
c.InteractiveShellApp.exec_lines = ['%autoreload 2']
c.InteractiveShellApp.exec_lines = [
    'from analyst import Analyst',
    'import scrapers'
]
