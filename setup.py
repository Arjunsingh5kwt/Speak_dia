from setuptools import setup, find_packages

setup(
    name='voice_processing_app',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.5',
        'torch>=1.10.0',
        # Add other dependencies
    ],
    python_requires='>=3.8,<3.11',
)