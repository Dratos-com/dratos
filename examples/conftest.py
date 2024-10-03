"""
! NOT AN EXAMPLE

What this script does:
Defines the server fixture for the tests.
"""

import subprocess
import pytest
import time
import os

# Project root path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['IS_TEST_ENV'] = 'true'
print("\033[94mTEST ENV STARTED\033[0m")

@pytest.fixture(scope="session", autouse=True)
def server_setup(request):
    server_file = os.path.join(root, 'test_server', 'server.py')
    proc = subprocess.Popen(['python', server_file])

    os.environ['TEST_API_BASE_URL'] = 'http://127.0.0.1:4000'
    
    time.sleep(2)

    def server_teardown():
        proc.terminate()
        proc.wait()
        os.environ['IS_TEST_ENV'] = 'false'
        print("\033[94mTEST ENV STOPPED\033[0m")

    request.addfinalizer(server_teardown)

@pytest.fixture(scope="session")
def api_url():
    return os.getenv('TEST_API_BASE_URL', 'http://127.0.0.1:4000')