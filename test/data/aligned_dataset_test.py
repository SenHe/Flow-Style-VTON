import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import linecache

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.fine_height=256
        self.fine_width=192

        self.text = './test_pairs.txt'

        dir_I = '_img'
        self.dir_I = os.path.join(opt.dataroot, opt.phase + dir_I)

        dir_C = '_clothes'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)

        dir_E = '_edge'
        self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)

        self.im_name = []
        self.c_name = []
        self.e_name = []
        self.get_file_name()
        #import ipdb; ipdb.set_trace()
        self.dataset_size = len(self.im_name)

    def get_file_name(self):

        with open(self.text, 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                self.im_name.append(os.path.join(self.dir_I, im_name))
                self.c_name.append(os.path.join(self.dir_C, c_name))
                self.e_name.append(os.path.join(self.dir_E, c_name))

    def __getitem__(self, index):        

        #file_path ='demo.txt'
        #im_name, c_name = linecache.getline(file_path, index+1).strip().split()

        I_path = os.path.join(self.im_name[index])
        I = Image.open(I_path).convert('RGB')

        params = get_params(self.opt, I.size)
        transform = get_transform(self.opt, params)
        transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        I_tensor = transform(I)
        #import ipdb; ipdb.set_trace()
        C_path = os.path.join(self.c_name[index])
        #print(self.c_name[index])
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform(C)

        E_path = os.path.join(self.e_name[index])
        E = Image.open(E_path).convert('L')
        E_tensor = transform_E(E)

        input_dict = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor, 'p_name':self.im_name[index].split('/')[-1]}
        return input_dict

    def __len__(self):
        return self.dataset_size 

    def name(self):
        return 'AlignedDataset'
