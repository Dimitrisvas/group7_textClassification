from setuptools import find_packages, setup

setup(
    name='flask_implementation',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True, # static & template directories
    zip_safe=False,
    install_requires=[
        'flask',
    ],
)