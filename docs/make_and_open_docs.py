import subprocess
import os
import sys
import webbrowser
from pathlib import Path

path_makefile = Path(__file__).resolve().parents[1] / "docs"

new_env = os.environ.copy()
new_env["SPHINXBUILD"] = 'D:\\algan\\.venv\\Scripts\\sphinx-build.exe'
subprocess.run(["make", "html"], shell=True, env=new_env)#, cwd=path_makefile)

website = (path_makefile / "build" / "html" / "index.html").absolute().as_uri()
try:  # Allows you to pass a custom browser if you want.
    webbrowser.get(sys.argv[1]).open_new_tab(f"{website}")
except IndexError:
    webbrowser.open_new_tab(f"{website}")
