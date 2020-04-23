"""
Contains functions and objects for reading a subset of a SWIFT dataset and writing
it to a new file.
"""

from swiftsimio.reader import SWIFTUnits, SWIFTMetadata
from swiftsimio.accelerated import read_ranges_from_file

import unyt
import h5py
import numpy as np

class SWIFTDatasubset(object):
    """
    class for reading in subset of SWIFT snapshot
    """

    def __init__(self, filename: str, mask: SWIFTMask, mask_size: int):
        # Load the units and metadata
        self.filename = filename
        self.mask = mask
        self.mask_size = mask_size
        self.dataset_names = []
        self.datasets = []
        
        with h5py.File(filename, "r") as infile:
            # Find all the datasets in the snapshot
            infile.visititems(self.find_datasets)

    def find_datasets(self, name, node):
        if isinstance(node, h5py.Dataset) and not name in self.dataset_names:
            self.dataset_names.append(name)

    def load_datasets(self):
        with h5py.File(self.filename, "r") as infile:
            if self.mask is not None:
                # ALEXEI: note we restrict the range of dataset names for testing for now
                for name in self.dataset_names[0:2]:
                    # get output dtype and size 
                    first_value = infile[name][0]
                    input_type = first_value.dtype
                    input_size = first_value.size
                    if output_size != 1:
                        output_shape = (mask_size, output_size)
                    else:
                        output_shape = mask_size

                    self.datasets.append(read_ranges_from_file(infile[name], mask, output_shape = output_shape, output_type = output_type))



class SWIFTWriterDatasubset(object):
    """
    Class for writing subset of SWIFT snapshot
    """

    def __init__(self, infile: str, outfile: str):
        self.input_filename = infile
        self.output_filename = outfile

        #self.subset = SWIFTDatasubset(self.input_filename)
        self.write_metadata()

    def write_metadata(self):
        """
        Copy over all the metadata from snapshot to output file

        ALEXEI: rewrite this taking advantage of finding the 
        datasets in SWIFTDatasubset
        """
        with h5py.File(self.input_filename, "r") as input_file:
            with h5py.File(self.output_filename, "w") as output_file:
                for field in input_file.keys():
                    if not any([group_str in field for group_str in ["PartType", "Cells"]]):
                        print("copying "+field)
                        input_file.copy(field, output_file)
