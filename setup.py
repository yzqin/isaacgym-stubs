from setuptools import setup

setup(
    name='isaacgym-stubs',
    author="Yuzhe Qin",
    author_email="y1qin@ucsd.edu",
    description="PEP 561 type stubs for isaacgym.gymapi",
    version='1.0rc4',
    packages=['isaacgym-stubs'],
    install_requires=['isaacgym>=1.0rc4'],
    package_data={"isaacgym-stubs": ['gymapi.pyi', '__init__.pyi']},
)
