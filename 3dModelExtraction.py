import pyvista as pv
import numpy as np
#import cv2
#import matplotlib.pyplot as plt

#splash to gridstream dump001 --npix=100,100,000 --grid=6
#set splash.limits too

# File path
fname ="/Users/nicoleberchtold/runs/disc_flyby/flyby_00056.gridstream"

with open(fname, "rb") as f:
    # Read 4 int32s: nx, ny, nz, ncolumns
    header_ints = np.fromfile(f, dtype=np.int32, count=4)
    nx, ny, nz, ncolumns = header_ints
    print(f"Grid size: {nx} x {ny} x {nz}, Columns: {ncolumns}")
    
    # Read 7 float64s: time, xmin, xmax, ymin, ymax, zmin, zmax
    header_floats = np.fromfile(f, dtype=np.float64, count=7)
    time, xmin, xmax, ymin, ymax, zmin, zmax = header_floats
    print(f"Time: {time}, X: {xmin}–{xmax}, Y: {ymin}–{ymax}, Z: {zmin}–{zmax}")
    
    # Read density values
    n_vals = nx * ny * nz
    rho = np.fromfile(f, dtype=np.float64, count=n_vals)
    
    # Reshape to 3D array (z, y, x) layout
    rho = rho.reshape((nz, ny, nx))

rho = rho.transpose(2, 1, 0)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
z = np.linspace(zmin, zmax, nz)

print('data extracted')

'''for i in range(-10, 10):
    im = rho[:, :, len(z)//2 - 3]
    im = np.log(im)
    im = im - im.min()       # make all values ≥ 0
    im = im / im.max()       # scale to [0, 1]
    im = (im * 255).astype(np.uint8)
    cv2.imshow('im', im)
    cv2.waitKey()
'''

#from pyvista import examples
rho_log = np.log(rho)
rho_log -= rho_log.min()
rho_log /= rho_log.max() #norm density

nx, ny, nz = rho_log.shape

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)
dz = (zmax - zmin) / (nz - 1)

grid = pv.ImageData(
    dimensions=(nx, ny, nz),
    spacing=(dx, dy, dz),
    origin=(xmin, ymin, zmin),
)

# Add scalar field
grid["density"] = rho_log.flatten(order="F")  # Fortran order (z, y, x)

density = [0.9, 0.95, 0.99]
contour = grid.contour(isosurfaces=density) #get surface at which density = this
contour.save(f"/Users/nicoleberchtold/Desktop/astroPhys/output_{density}.stl")
print('saved')
# Plot
plotter = pv.Plotter()
plotter.add_mesh(contour, color="orange", opacity=0.6)
plotter.show()