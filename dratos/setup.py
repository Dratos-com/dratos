from setuptools import setup, find_packages

setup(
    name='memento',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here
        # e.g., 'requests', 'numpy', etc.
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts here
            # e.g., 'your_command=your_module:main_function',
        ],
    },
)
