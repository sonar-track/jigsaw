from setuptools import setup, find_namespace_packages


setup(
    name='sonar.jigsaw',
    version='0.0.1',
    author='Phong D. Vo',
    packages=find_namespace_packages(include=['sonar.*']),
    scripts=[
        'scripts/jigsaw.py'
    ],
    requires=[
        'argparse',
        'yaml',
        'tensorflow'
    ]
)