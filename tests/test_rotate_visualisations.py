from tests.helper import requires
from swiftsimio.subset_writer import find_links, write_metadata
from swiftsimio import mask, cosmo_array, load
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.slice import slice_gas
from swiftsimio.visualisation.volume_render import render_gas
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
import h5py
from numpy import array_equal, mean

@requires("cosmological_volume.hdf5")
def test_rotation(filename):
    output = "single_particle.hdf5"
    
    data_mask = mask(filename)
    boxsize = data_mask.metadata.boxsize
    region = [[0, b] for b in boxsize]
    data_mask.constrain_spatial(region)
    
    # Write the metadata
    infile = h5py.File(filename, "r")
    outfile = h5py.File(output, "w")
    list_of_links, _ = find_links(infile)
    write_metadata(infile, outfile, list_of_links, data_mask)
    
    # Write a single particle
    particle_coords = cosmo_array([[1,1,1], [1,1,1]], data_mask.metadata.units.length)
    particle_masses = cosmo_array([1, 1], data_mask.metadata.units.mass)
    mean_h = mean(infile["/PartType0/SmoothingLengths"])
    particle_h = cosmo_array([mean_h, mean_h], data_mask.metadata.units.length)
    particle_ids = [1, 2]
    
    coords = outfile.create_dataset("/PartType0/Coordinates", data=particle_coords)
    for name, value in infile["/PartType0/Coordinates"].attrs.items():
        coords.attrs.create(name, value)
    
    masses = outfile.create_dataset("/PartType0/Masses", data=particle_masses)
    for name, value in infile["/PartType0/Masses"].attrs.items():
        masses.attrs.create(name, value)
    
    h = outfile.create_dataset("/PartType0/SmoothingLengths", data=particle_h)
    for name, value in infile["/PartType0/SmoothingLengths"].attrs.items():
        h.attrs.create(name, value)
    
    ids = outfile.create_dataset("/PartType0/ParticleIDs", data=particle_ids)
    for name, value in infile["/PartType0/ParticleIDs"].attrs.items():
        ids.attrs.create(name, value)
    
    outfile.create_dataset("/PartType1/Coordinates", shape = particle_coords.shape)
    
    # Tidy up
    infile.close()
    outfile.close()
    
    # Start from the beginning, open the file 
    data = load(output)
    
    # Compute rotation matrix for rotating around particle
    centre = particle_coords[0]
    rotate_vec = [0.5,0.5,0.5]
    matrix = rotation_matrix_from_vector(rotate_vec, axis = 'z')
    
    # Check the projection first
    unrotated = project_gas(
        data, 
        resolution=1024, 
        project="masses", 
        parallel=True
    )
    
    rotated = project_gas(
        data, 
        resolution=1024, 
        project="masses", 
        rotation_center = centre, 
        rotation_matrix = matrix,
        parallel=True
    )
    
    assert(array_equal(rotated, unrotated))
    
    # Now check rotations in slices
    # First find the locations of the slices that contain our particle
    z_range = boxsize[2]
    slice_z = particle_coords[0,2]/z_range
    
    unrotated = slice_gas(
        data, 
        resolution=1024, 
        slice=slice_z,
        project="masses", 
        parallel=True
    )
    
    rotated = slice_gas(
        data, 
        resolution=1024, 
        slice=slice_z,
        project="masses", 
        rotation_center = centre, 
        rotation_matrix = matrix,
        parallel=True
    )
    
    # Check that we didn't miss the particle
    assert(unrotated.any())
    assert(rotated.any())
    
    assert(array_equal(rotated, unrotated))
    
    # And now check the volume render
    unrotated = render_gas(
        data, 
        resolution=256, 
        project="masses", 
        parallel=True
    )
    
    rotated = render_gas(
        data, 
        resolution=256, 
        project="masses", 
        rotation_center = centre, 
        rotation_matrix = matrix,
        parallel=True
    )
    
    assert(array_equal(rotated, unrotated))
