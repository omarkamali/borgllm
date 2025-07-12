import os
import subprocess
import pytest


def test_docs_build():
    """Test that the Sphinx documentation can be built successfully."""
    # Ensure we are in the project root
    original_cwd = os.getcwd()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)

    try:
        # Run the hatch docs command
        result = subprocess.run(
            ["hatch", "run", "docs"], capture_output=True, text=True, check=False
        )

        # Assert the command exited successfully
        assert result.returncode == 0, f"Docs build failed with errors: {result.stderr}"

        # Assert that the build output contains success message and no critical warnings/errors
        assert (
            "build succeeded" in result.stdout.lower()
            or "build succeeded" in result.stderr.lower()
        ), f"'build succeeded' message not found in output. Stdout: {result.stdout}, Stderr: {result.stderr}"

        # Check for specific critical warnings or errors that should fail the build
        # (e.g., severe Sphinx errors, unhandled exceptions)
        assert (
            "error" not in result.stderr.lower()
        ), f"Critical errors found during docs build: {result.stderr}"
        assert (
            "exception" not in result.stderr.lower()
        ), f"Exceptions found during docs build: {result.stderr}"

        # Verify that the main HTML file was created
        docs_build_path = os.path.join(project_root, "docs", "build", "index.html")
        assert os.path.exists(
            docs_build_path
        ), f"Docs build output file not found: {docs_build_path}"

    finally:
        # Change back to the original working directory
        os.chdir(original_cwd)
