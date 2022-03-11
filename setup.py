from setuptools import setup, find_packages

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

setup(
    name='bagofmaml',
    version='0.1.0',
    author="Marc Rußwurm",
    author_email="marc.russwurm@epfl.ch",
    packages=find_packages(include=['bagofmaml', 'bagofmaml.*']),
    install_requires=requirements,
    extras_require={"examples": ["numpy", "pandas", "matplotlib"]}
)
