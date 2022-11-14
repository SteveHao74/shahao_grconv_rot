import glob
import numpy as np
dict_root = '/media/abb/Data/Data/ggcnn_rot_data/all_data/'
npz_file = glob.glob(dict_root+'*.npz')
# print((npz_file))

npz = np.load(npz_file[0])

print(npz['tags'])