from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="HideNSeek",
    version="0.0.1",
    author="Dmitri Iourovitski, LachsBagel",
    author_email="ADD DMITRI's EMAIL HERE, lachsbagel@proton.me",
    description="An algorithm to verify whether "
                "someone who's hosting an LLM is really hosting the LLM they claim they are.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MorpheusAIs/HideNSeek",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
)
