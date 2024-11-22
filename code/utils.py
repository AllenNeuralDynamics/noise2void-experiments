import os
import numpy as np
import tifffile


def tile_and_save(array, name, tile_size=(64, 64, 64), output_dir='tiles', filename_pattern='tile_{z}_{y}_{x}.tiff'):
    """
    Tiles a 3D NumPy array and saves each tile as a separate TIFF file.

    Parameters:
    - array: 3D NumPy array to be tiled.
    - tile_size: Tuple of ints (depth, height, width) specifying the size of each tile.
    - output_dir: Directory where the TIFF files will be saved.
    - filename_pattern: Pattern for naming the TIFF files.
      Use '{z}', '{y}', and '{x}' as placeholders for tile indices.

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    D, H, W = array.shape
    d, h, w = tile_size
    
    num_tiles_z = int(np.ceil(D / d))
    num_tiles_y = int(np.ceil(H / h))
    num_tiles_x = int(np.ceil(W / w))
    
    for z in range(num_tiles_z):
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                start_z = z * d
                end_z = min((z + 1) * d, D)
                start_y = y * h
                end_y = min((y + 1) * h, H)
                start_x = x * w
                end_x = min((x + 1) * w, W)

                tile = array[start_z:end_z, start_y:end_y, start_x:end_x]
                
                filename = filename_pattern.format(n=name, z=z, y=y, x=x)
                filepath = os.path.join(output_dir, filename)
                
                tifffile.imwrite(filepath, tile)
                