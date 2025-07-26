# Testing Guide

This directory contains some automated tests for the project, 
and you are welcome and encoraged to add more.

See the [`pytest`Getting Started docs](https://docs.pytest.org/en/stable/getting-started.html#getstarted) for help with writing
your own test scripts.

## 🚀 How to Run Tests

1.  **First, install `pytest`:**
    ```bash
    pip install pytest
    ```

2.  **Then execute run one of these commands from the _project root_:**
    ```bash
    pytest # runs all discovered tests
    pytest tests/my_test.py # runs only tests(s) you specify
    ```

`pytest` will automatically discover and run _all_ test files, unless 
speific file(s) are specified. 
The `pytest.ini` file in the root directory is configured to ensure 
that imports from `src/` work correctly. 
