from setuptools import setup, find_packages

setup(
    name="clip_debias",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "ftfy",
        "regex",
        "tqdm",
        "numpy",
        "Pillow",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "open-clip-torch",
    ],
)
