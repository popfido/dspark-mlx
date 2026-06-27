import os
import sys

# Make `import dspark_mlx` resolve when running pytest from any cwd, without
# requiring an editable install into the host venv.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
