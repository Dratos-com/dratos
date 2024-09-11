from setuptools import setup, find_packages

setup(
    name="YourPackageName",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourgithubrepo",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pyarrow",
        "pydantic",
        "mlflow",
        "wandb",
        "ray",
        "ulid-py",
        "uuid",
        "unittest",
        "json",
        "weave",  # Assuming 'weave' is available as a package
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
