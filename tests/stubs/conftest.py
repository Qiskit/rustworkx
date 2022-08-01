# Configure Pytest to ignore stub tests in early versions
import sys

collect_ignore_glob = []
if sys.version_info < (3, 8):
    collect_ignore_glob.append("*.py")
