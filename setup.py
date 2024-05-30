from setuptools import setup

setup(
    name="DDDM",
    py_modules=["DDDM"],
    install_requires=["blobfile>=1.0.5", "torch","tqdm","lpips"],
)
