from setuptools import find_packages, setup

setup(
    name='postamats',
    version='1.0.0',
    description='core module for hackatone https://leaders2022.innoagency.ru/task10.html',
    author='Optimists',
    packages=find_packages('src'),
    package_dir={'': 'src'}
)
