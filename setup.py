import setuptools
from swiftsimio import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="swiftsimio",
    version=__version__,
    description="SWIFTsim (swift.dur.ac.uk) i/o routines for python.",
    url="https://github.com/swiftsim/swiftsimio",
    author="Josh Borrow",
    author_email="joshua.borrow@durham.ac.uk",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    scripts=["swiftsnap"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "unyt>=2.3.0", "h5py"],
)
