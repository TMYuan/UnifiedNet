import os
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision
from glob import glob
from .flow_utils import *

def file_utils(root, dataset, dstype, iext):
    """
    filename type
    ------------------------------------
    FlyingChairs:
        flow name type : 00000_flow.flo
        img name type : 00000_img1.ppm
    ------------------------------------
    MPISintel:
        flow name type : frame0000.flo
        img name type : frame0000.png    
    ------------------------------------
    others:
        flow name type : frame0000.flo
        img name type : frame0000.jpg    
    ------------------------------------           
    """
    flow_root = os.path.join(root, 'flow')
    image_root = os.path.join(root, dstype)
#     print(flow_root)
#     print(image_root)
    flow_list = []
    image_list = []
    
    if dataset == 'FlyingChairs':
        file_list = sorted(glob(flow_root + '/*.flo'))        
#         print(file_list)
        for file in file_list:        
            fbase = file[len(flow_root)+1:]
            fprefix = fbase[-9:-4]
            fnum = int(fbase[:5])
            img = os.path.join(image_root, "%05d"%fnum + "_img1" + '.ppm')
            if not os.path.isfile(img) or not os.path.isfile(file):
                continue
            image_list += [img]
            flow_list += [file]
        
    elif dataset == 'MPISintel':    
        file_list = sorted(glob(os.path.join(flow_root, '*/*.flo')))        
        for file in file_list:        
            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])
            img = os.path.join(image_root, fprefix + "%04d"%fnum + '.png')       
            if not os.path.isfile(img) or not os.path.isfile(file):
                continue
            image_list += [img]
            flow_list += [file]
        
    elif dataset == 'others':
        file_list = sorted(glob(os.path.join(flow_root, '*.flo')))        
        for file in file_list:        
            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])
            img = os.path.join(image_root, fprefix + "%04d"%fnum + iext)       
            if not os.path.isfile(img) or not os.path.isfile(file):
                continue
            image_list += [img]
            flow_list += [file]
#     print(image_list)
#     print(flow_list)
        
    return image_list, flow_list

class Datasets(data.Dataset):
    def __init__(self, root = '',dataset='others', dstype='image', iext='.jpg', img_size = 56, transforms = None):
        self.transforms = transforms
        self.img_size = img_size
        self.image_list, self.flow_list = file_utils(root, dataset, dstype, iext)            
        self.size = len(self.image_list)
        assert (len(self.image_list) == len(self.flow_list))
        
    def __getitem__(self, index):
        index = index % self.size
        img = cv2.imread(self.image_list[index])
        flow = readFlow(self.flow_list[index]).astype('float32')
#         flow = np.transpose(flow.astype(np.float32), (1, 2, 0))

        img = cv2.resize(img, (self.img_size,self.img_size))
        flow = cv2.resize(flow, (self.img_size,self.img_size))
        
        if self.transforms is not None:
            img = self.transforms(img)
            flow = self._flow_trans(flow)
        return img, flow
    
    def __len__(self):
        return self.size
    
    def _flow_trans(self, flow):
        # To tensor, normalize, change channel
        # 1. normalize
        flow[..., 0] = flow[..., 0] / 20.0
        flow[..., 1] = flow[..., 1] / 20.0
        
        # 2. change channel
        flow = np.moveaxis(flow, -1, 0)
        
        # 3. To tensor
        flow = torch.from_numpy(flow)
        return flow
    
