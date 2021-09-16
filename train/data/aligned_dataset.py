import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw
import random


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.diction={}

        if opt.isTrain or opt.use_encoded_image:
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))

        self.fine_height=256
        self.fine_width=192
        self.radius=5

        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)

        if opt.isTrain or opt.use_encoded_image:
            dir_E = '_edge'
            self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
            self.E_paths = sorted(make_dataset(self.dir_E))

        if opt.isTrain or opt.use_encoded_image:
            dir_C = '_color'
            self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)
            self.C_paths = sorted(make_dataset(self.dir_C))


    def __getitem__(self, index):        

        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('L')

        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(B)

        C_path = self.C_paths[index]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_B(C)

        E_path = self.E_paths[index]
        E = Image.open(E_path).convert('L')
        E_tensor = transform_A(E)

        index_un = np.random.randint(14221)
        C_un_path = self.C_paths[index_un]
        C_un = Image.open(C_un_path).convert('RGB')
        C_un_tensor = transform_B(C_un)

        E_un_path = self.E_paths[index_un]
        E_un = Image.open(E_un_path).convert('L')
        E_un_tensor = transform_A(E_un)

        pose_name =B_path.replace('.png', '_keypoints.json').replace('.jpg','_keypoints.json').replace('train_img','train_pose')
        with open(osp.join(pose_name), 'r') as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label['people'][0]['pose_keypoints']
            except IndexError:
                pose_data = [0 for i in range(54)]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = transform_B(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        P_tensor=pose_map

        densepose_name = B_path.replace('.png', '.npy').replace('.jpg','.npy').replace('train_img','train_densepose')
        dense_mask = np.load(densepose_name).astype(np.float32)
        dense_mask = transform_A(dense_mask)

        if self.opt.isTrain:
            input_dict = { 'label': A_tensor, 'image': B_tensor, 'path': A_path, 'img_path': B_path ,'color_path': C_path,'color_un_path': C_un_path,
                            'edge': E_tensor, 'color': C_tensor, 'edge_un': E_un_tensor, 'color_un': C_un_tensor, 'pose':P_tensor, 'densepose':dense_mask
                          }

        return input_dict

    def __len__(self):
        return len(self.A_paths) // (self.opt.batchSize * self.opt.num_gpus) * (self.opt.batchSize * self.opt.num_gpus)

    def name(self):
        return 'AlignedDataset'

class AlignedDataset_aug(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.diction={}

        self.shift = [25, 45, 65]

        self.scale = [1.25, 1.5, 1.75]

        if opt.isTrain or opt.use_encoded_image:
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))

        self.fine_height=256
        self.fine_width=192
        self.radius=5

        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)

        if opt.isTrain or opt.use_encoded_image:
            dir_E = '_edge'
            self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
            self.E_paths = sorted(make_dataset(self.dir_E))

        if opt.isTrain or opt.use_encoded_image:
            dir_C = '_color'
            self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)
            self.C_paths = sorted(make_dataset(self.dir_C))


    def __getitem__(self, index):  



        p_aug = random.random()


        A_path = self.A_paths[index]

        if p_aug<0.33:

            temp_A = Image.open(A_path).convert('L')

        elif p_aug<0.66:

            A = np.asarray(Image.open(A_path).convert('L'))
            temp = np.zeros((A.shape[0], A.shape[1]))
            temp = np.uint8(temp)

            shift = self.shift[random.randrange(0,3)]
            temp[shift:A.shape[0],:] = A[:A.shape[0]-shift,:]
            temp_A = Image.fromarray(temp).convert('L')

            
        else:

            A = np.asarray(Image.open(A_path).convert('L'))
            scale = self.scale[random.randrange(0,3)]
            temp = np.zeros((int(scale*A.shape[0]), int(scale*A.shape[1])))
            temp = np.uint8(temp)

            temp[(int(scale*A.shape[0])-A.shape[0])//2:(int(scale*A.shape[0])-A.shape[0])//2+A.shape[0],(int(scale*A.shape[1])-A.shape[1])//2:(int(scale*A.shape[1])-A.shape[1])//2+A.shape[1]] = A
            temp_A = Image.fromarray(temp).convert('L')
            temp_A = temp_A.resize((A.shape[1],A.shape[0]),Image.NEAREST)


        params = get_params(self.opt, temp_A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(temp_A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(temp_A) * 255.0

        B_path = self.B_paths[index]

        if p_aug<0.33:

            temp_B = Image.open(B_path).convert('RGB')

        elif p_aug<0.66:

            B = np.asarray(Image.open(B_path).convert('RGB'))
            temp = np.ones((B.shape[0], B.shape[1], 3))
            temp[:,:,0] = temp[:,:,0] * B[10,10,0]
            temp[:,:,1] = temp[:,:,1] * B[10,10,1]
            temp[:,:,2] = temp[:,:,2] * B[10,10,2]
            temp = np.uint8(temp)

            #shift = self.shift[random.randrange(0,3)]
            temp[shift:B.shape[0],:,:] = B[:B.shape[0]-shift,:,:]
            temp_B = Image.fromarray(temp).convert('RGB')

            
        else:

            B = np.asarray(Image.open(B_path).convert('RGB'))
            #scale = self.scale[random.randrange(0,3)]
            temp = np.ones((int(scale*B.shape[0]), int(scale*B.shape[1]), 3))
            temp[:,:,0] = temp[:,:,0] * B[10,10,0]
            temp[:,:,1] = temp[:,:,1] * B[10,10,1]
            temp[:,:,2] = temp[:,:,2] * B[10,10,2]
            temp = np.uint8(temp)

            temp[(int(scale*B.shape[0])-B.shape[0])//2:(int(scale*B.shape[0])-B.shape[0])//2+B.shape[0],(int(scale*B.shape[1])-B.shape[1])//2:(int(scale*B.shape[1])-B.shape[1])//2+B.shape[1], :] = B
            temp_B = Image.fromarray(temp).convert('RGB')
            temp_B = temp_B.resize((B.shape[1],B.shape[0]))

        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(temp_B)

        C_path = self.C_paths[index]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_B(C)

        E_path = self.E_paths[index]
        E = Image.open(E_path).convert('L')
        E_tensor = transform_A(E)

        index_un = np.random.randint(14221)
        C_un_path = self.C_paths[index_un]
        C_un = Image.open(C_un_path).convert('RGB')
        C_un_tensor = transform_B(C_un)

        E_un_path = self.E_paths[index_un]
        E_un = Image.open(E_un_path).convert('L')
        E_un_tensor = transform_A(E_un)

        pose_name =B_path.replace('.png', '_keypoints.json').replace('.jpg','_keypoints.json').replace('train_img','train_pose')
        with open(osp.join(pose_name), 'r') as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label['people'][0]['pose_keypoints']
            except IndexError:
                pose_data = [0 for i in range(54)]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')

            if p_aug<0.33:

                temp_one_map = one_map

            elif p_aug<0.66:

                one_map = np.asarray(one_map)
                temp = np.zeros((one_map.shape[0], one_map.shape[1]))
                temp = np.uint8(temp)

                temp[shift:A.shape[0],:] = one_map[:A.shape[0]-shift,:]
                temp_one_map = Image.fromarray(temp).convert('L')

            
            else:

                one_map = np.asarray(one_map)
                temp = np.zeros((int(scale*one_map.shape[0]), int(scale*one_map.shape[1])))
                temp = np.uint8(temp)

                temp[(int(scale*one_map.shape[0])-one_map.shape[0])//2:(int(scale*one_map.shape[0])-one_map.shape[0])//2+one_map.shape[0],(int(scale*one_map.shape[1])-one_map.shape[1])//2:(int(scale*one_map.shape[1])-one_map.shape[1])//2+one_map.shape[1]] = one_map
                temp_one_map = Image.fromarray(temp).convert('L')
                temp_one_map = temp_one_map.resize((one_map.shape[1],one_map.shape[0]),Image.NEAREST)
            one_map = transform_B(temp_one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        P_tensor=pose_map

        densepose_name = B_path.replace('.png', '.npy').replace('.jpg','.npy').replace('train_img','train_densepose')
        #dense_mask = np.load(densepose_name).astype(np.float32)

        if p_aug<0.33:

            temp_dense_mask = np.load(densepose_name).astype(np.float32)

        elif p_aug<0.66:

            dense_mask = np.load(densepose_name).astype(np.float32)
            temp = np.zeros((dense_mask.shape[0], dense_mask.shape[1])).astype(np.float32)

            temp[shift:dense_mask.shape[0],:] = dense_mask[:dense_mask.shape[0]-shift,:]
            temp_dense_mask = temp

            
        else:

            dense_mask = np.load(densepose_name).astype(np.float32)
            temp = np.zeros((int(scale*dense_mask.shape[0]), int(scale*dense_mask.shape[1]))).astype(np.float32)


            temp[(int(scale*dense_mask.shape[0])-dense_mask.shape[0])//2:(int(scale*dense_mask.shape[0])-dense_mask.shape[0])//2+dense_mask.shape[0],(int(scale*dense_mask.shape[1])-dense_mask.shape[1])//2:(int(scale*dense_mask.shape[1])-dense_mask.shape[1])//2+dense_mask.shape[1]] = dense_mask

            temp = Image.fromarray(np.uint8(temp))
            temp = temp.resize((dense_mask.shape[1],dense_mask.shape[0]),Image.NEAREST)
            temp_dense_mask = np.asarray(temp).astype(np.float32)

        dense_mask = transform_A(temp_dense_mask)

        if self.opt.isTrain:
            input_dict = { 'label': A_tensor, 'image': B_tensor, 'path': A_path, 'img_path': B_path ,'color_path': C_path,'color_un_path': C_un_path,
                            'edge': E_tensor, 'color': C_tensor, 'edge_un': E_un_tensor, 'color_un': C_un_tensor, 'pose':P_tensor, 'densepose':dense_mask
                          }

        return input_dict

    def __len__(self):
        return len(self.A_paths) // (self.opt.batchSize * self.opt.num_gpus) * (self.opt.batchSize * self.opt.num_gpus)

    def name(self):
        return 'AlignedDataset_aug'
