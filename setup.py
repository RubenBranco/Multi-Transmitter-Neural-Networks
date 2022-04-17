from setuptools import setup, find_packages

setup(
    name="multi-transmitter-nn",
    packages=find_packages(exclude=[]),
    version="0.0.1",
    description="Neural Networks based on multi-transmitter neurons.",
    author="Ruben Branco",
    author_email="rmbranco@fc.ul.pt",
    url="https://github.com/RubenBranco/Multi-Transmitter-Neural-Networks",
    keywords=[
        "artificial intelligence",
        "deep learning",
    ],
    install_requires=[
        "torch"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
    ],
)
