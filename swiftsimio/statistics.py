"""
Reader for the statistics file.
"""

import unyt
import regex as re

from typing import List, Dict

from swiftsimio.accelerated import list_of_strings_to_arrays


class SWIFTStatisticsFile(object):
    """
    SWIFT statistics files (e.g. SFR.txt, energy.txt) reader.
    """

    # Names from the header.
    header_names: List[str]
    # Units (unyt-based) from the header
    header_units: Dict[str, unyt.unyt_quantity]
    # snake_case names from the header
    header_snake_case_names: List[str]
    # Raw lines as strings, read from the file.
    raw_lines: List[str]

    def __init__(self, filename: str):
        """
        Parameters
        ----------

        filename: str
            File name for the statistics file.
        """

        self.filename = filename

        self._read_file()
        self._process_raw_lines()

        return

    def _read_file(self):
        """
        Reads the header of the file, including loading the units.
        """

        # Read the header and use custom regex parsing.

        with open(self.filename, "r") as handle:
            lines = handle.readlines()

        current_line = 0

        header_names = []
        header_units = {}
        current_name = None

        # Regex for matching
        regex_name = re.compile(r"# \(([0-9]*)\) +([^\.\n]*)")
        regex_unit = re.compile(r"# *Unit = ([^\s]+) ?(.*)")

        while lines[current_line].startswith("#"):
            # Regex match each line to see if it is a unit
            # or a name

            current_string = lines[current_line]
            current_line += 1

            name_match = regex_name.match(current_string)

            if name_match:
                current_name = name_match.group(2)
                header_units[current_name] = unyt.dimensionless
                header_names.append(current_name)

                continue

            unit_match = regex_unit.match(current_string)

            if unit_match:
                if unit_match.group(1) != "dimensionless":
                    header_units[current_name] = unyt.unyt_quantity(
                        float(unit_match.group(1)), unit_match.group(2)
                    )
                else:
                    header_units[current_name] = unyt.dimensionless

                continue

        # The last line will be the names, so extract those here.
        header_snake_case_names = [
            x.replace(".", "").replace(" ", "_").replace("\n", "").lower()
            for x in re.split(r"\s{2,}", lines[current_line - 1][1:])
            if x != ""
        ]

        self.header_names = header_names
        self.header_units = header_units
        self.header_snake_case_names = header_snake_case_names

        self.raw_lines = lines[current_line:]

        return

    def _process_raw_lines(self):
        """
        Processes the raw string lines read out of the header.
        """

        arrays = list_of_strings_to_arrays(lines=self.raw_lines)

        for array, header_name, header_snake_case_name in zip(
            arrays, self.header_names, self.header_snake_case_names
        ):
            setattr(
                self,
                header_snake_case_name,
                unyt.unyt_array(
                    array, units=self.header_units[header_name], name=header_name
                ),
            )

        return

    def __str__(self):
        return (
            f"Statistics file: {self.filename}, containing fields: "
            f"{', '.join(self.header_snake_case_names)}"
        )

    def __repr__(self):
        return str(self)
