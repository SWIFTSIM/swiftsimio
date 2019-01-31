import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="swiftsimio",
    version="0.1.0",
    description="SWIFTsim (swift.dur.ac.uk) i/o routines for python.",
    url="https://gitlab.cosma.dur.ac.uk/jborrow/SWIFTsimIO",
    author="Josh Borrow",
    author_email="joshua.borrow@durham.ac.uk",
    packages=["swiftsimio"],
    long_description=long_description,
    zip_safe=False,
        classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: LGPLv3 License",
        "Operating System :: OS Independent",
    ],
)

