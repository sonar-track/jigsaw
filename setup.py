from setuptools import setup, find_namespace_packages


setup(
    name='eyenav.jigsaw',
    version='0.0.1',
    author='Phong D. Vo',
    packages=find_namespace_packages(include=['eyenav.*']),
    scripts=[
        'scripts/jigsaw.py'
    ]
)