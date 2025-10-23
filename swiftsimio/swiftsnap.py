#!python3
"""
``swiftsnap``: check the metadata of a SWIFT snapshot easily from the command line.

See the -h invocation for more details.
"""

import argparse as ap

SCREEN_WIDTH = 80


def decode(bytestring: object) -> str:
    """
    Decode input if it is a bytestring.

    If the input is not a bytestring, we instead try to interpret it as a string.

    Parameters
    ----------
    bytestring : object
        The (possibly) bytes object.

    Returns
    -------
    str
       The resulting string.
    """
    try:
        return bytestring.decode("utf-8")
    except AttributeError:
        return str(bytestring)


parser = ap.ArgumentParser(
    prog="swiftsnap",
    description=(
        "Prints metadata to the console, read from the swift snapshots that "
        "are given. Includes cosmology, run, and output information."
    ),
    epilog=("Example usage:\n  swiftsnap output_0000.hdf5"),
)

parser.add_argument(
    "snapshots",
    metavar="Snapshots",
    type=str,
    nargs="+",
    help=(
        "Snapshots you wish to view metadata for. Supports standard "
        "globbing syntax, so you can do output_00{10..14}.hdf5."
    ),
)

parser.add_argument(
    "-r",
    "--redshift",
    required=False,
    default=False,
    action="store_true",
    help=(
        "Print out just the redshifts of the snapshots and exit. "
        "Useful if you want to use these values in a shell script."
    ),
)

parser.add_argument(
    "-a",
    "--scale-factor",
    required=False,
    default=False,
    action="store_true",
    help=(
        "Print out just the scale factor of the snapshots and exit. "
        "Useful if you want to use these values in a shell script."
    ),
)


def swiftsnap() -> None:
    """
    Access SWIFT snapshot metadata from the command line.

    For details see the command line argument parser help.
    """
    import swiftsimio as sw
    from swiftsimio.metadata.objects import _metadata_discriminator

    from swiftsimio.metadata.particle import particle_name_underscores
    from textwrap import wrap

    args = parser.parse_args()
    snapshots = args.snapshots

    # First, validate the snapshots.
    for filename in snapshots:
        try:
            # returns True or raises:
            sw.validate_file(filename)
        except IOError:
            print(f"{filename} is not a SWIFT snapshot.")
            exit(1)

    # Now that we know they are valid, we can load the metadata.
    units_list = [sw.SWIFTUnits(snap) for snap in snapshots]
    metadata_list = [
        _metadata_discriminator(snap, units)
        for snap, units in zip(snapshots, units_list)
    ]

    if args.redshift:
        redshifts = [f"{snap.z:.4g}" for snap in metadata_list]
        print("\n".join(redshifts))
        exit(0)

    if args.scale_factor:
        scale_factors = [f"{snap.a:.4g}" for snap in metadata_list]
        print("\n".join(scale_factors))
        exit(0)

    # If we have not activated special modes, now it's time to get going with the
    # printing of metadata!

    for data in metadata_list:
        # Snapshot state information
        print(f"{decode(data.run_name)}")
        output_string = f"Written at: {data.snapshot_date}"
        print(f"{output_string}")
        activated_policies = [
            policy_name for policy_name, active in data.policy.items() if active
        ]
        policy_strings = wrap(
            f"Active policies: {', '.join(activated_policies)}", SCREEN_WIDTH
        )
        for policy_string in policy_strings:
            print(policy_string)

        print()
        output_type_string = (
            f"Output type: {data.output_type}, Output selection: {data.select_output}"
        )
        print(f"{output_type_string}")
        print()

        numbers_of_particles = "Number of particles: " + ", ".join(
            [
                f"{x.title().replace('_', ' ')}: {getattr(data, f'n_{x}', 0):.4g}"
                for x in particle_name_underscores.values()
            ]
        )
        numbers_of_particles_strings = wrap(numbers_of_particles, SCREEN_WIDTH)
        for numbers_of_particles_string in numbers_of_particles_strings:
            print(numbers_of_particles_string)

        print()

        # Code configuration

        print(data.compiler_info)
        print(data.code_info)

        print()

        # Current state information
        time_string = (
            f"Simulation state: z={data.z:.4g}, a={data.a:.4g}, t={data.time:.4g}"
        )
        print(f"{time_string}")
        print()
        print(f"Cosmology: {data.cosmology}")
        print()

        # Physics information
        print(f"Gravity scheme: {decode(data.gravity_scheme.get('Scheme', b'None'))}")

        print(
            f"Hydrodynamics scheme: {decode(data.hydro_scheme.get('Scheme', b'None'))}"
        )

        for name, scheme in data.subgrid_scheme.items():
            if name != "NamedColumns":
                print(f"{name.replace('Model', 'model')}: {decode(scheme)}")


if __name__ == "__main__":
    swiftsnap()
