import h5py
import numpy as np


partition = "testing"
datapath = f"/home/marc/Downloads/{partition}.h5"

# show the variables in selected file
fid = h5py.File(datapath,'r')

s1 = np.array(fid['sen1'])
