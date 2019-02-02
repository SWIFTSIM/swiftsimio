import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="swiftsimio",
    version="0.1.2",
    description="SWIFTsim (swift.dur.ac.uk) i/o routines for python.",
    url="https://gitlab.cosma.dur.ac.uk/jborrow/SWIFTsimIO",
    author="Josh Borrow",
    author_email="joshua.borrow@durham.ac.uk",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
        classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'unyt',
        'h5py'
    ]
)

