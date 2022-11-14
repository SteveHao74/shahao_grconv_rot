import torch

path = "/home/shahao/Project/ggcnn_rot/output/models/211216_1034_ggrot_gmd_wid/epoch_95_iou_0.67"
model=torch.load(path)
torch.save(model,"/home/shahao/Project/ggcnn_rot/output/models/transform/epoch_95_iou_0.67",_use_new_zipfile_serialization=False)