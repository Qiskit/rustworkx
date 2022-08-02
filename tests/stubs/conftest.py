# Configure Pytest to ignore stub tests in early versions
import sys

collect_ignore_glob = []
if sys.version_info < (3, 12):
    collect_ignore_glob.append("*.py")


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0  # ignore no tests found, it's intentional
