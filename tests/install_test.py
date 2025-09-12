import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

# Try importing the LAMMPS library
from lammps import lammps
print("LAMMPS Python module imported successfully!")

# Now try importing the OVITO library
from ovito.io import import_file
print("OVITO Python module imported successfully!")
