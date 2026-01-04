from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rag-prover",
    version="0.1.0",
    author="Ritik Jain",
    author_email="rjain92682@gmail.com",
    description="A Retrieval Augmented Generation (RAG) stack built atop DeepSeek-Prover-V2 7B",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rjain2470/rag-prover",
    packages=find_packages(where="src"),  # Look for packages in the 'src' directory
    package_dir={"": "src"},  # Root package directory is 'src'
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
)
