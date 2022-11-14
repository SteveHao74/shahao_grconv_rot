import os
import glob
file_path = '/media/abb/Data/Project/grasping-object-dataset/shapenet_4-sdf-object_size-1/'
save_path = '/media/abb/Data/Project/grasping-object-dataset/shapenet_4/'
obj_name = os.listdir(file_path)


for name in obj_name:
    obj_path = file_path + name + '/meshes/'
    obj_path = glob.glob(os.path.join(obj_path,'*.obj'))
    # print(obj_)
    # break
    os.system('cp ' + obj_path[0] + ' ' + save_path + name +'.obj')