import subprocess
import sys


def test_environment_check_is_non_blocking_without_require_flags():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.check_environment", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert '"python"' in result.stdout
    assert '"gpu"' in result.stdout
    assert '"docker"' in result.stdout

