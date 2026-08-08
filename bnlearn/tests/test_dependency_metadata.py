from pathlib import Path

import tomllib


PGMPY_CPP_REQUIREMENT = "pgmpy @ git+https://github.com/EdinhoAndra/pgmpy-cpp.git"


def test_pgmpy_dependency_points_exclusively_to_pgmpy_cpp():
    repository = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads(
        (repository / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = metadata["project"]["dependencies"]
    requirements = (
        (repository / "requirements.txt").read_text(encoding="utf-8").splitlines()
    )

    pgmpy_metadata = [
        dependency for dependency in dependencies if dependency.startswith("pgmpy @ ")
    ]
    pgmpy_requirements = [
        requirement
        for requirement in requirements
        if requirement.startswith("pgmpy @ ")
    ]

    assert pgmpy_metadata == [PGMPY_CPP_REQUIREMENT]
    assert pgmpy_requirements == [PGMPY_CPP_REQUIREMENT]
    assert all("EdinhoAndra/pgmpy.git" not in dependency for dependency in dependencies)
