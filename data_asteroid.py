import torch
import pickle
import numpy as np
import os
from skimage import io
from stl import mesh

class CustomDatasetFolder(torch.utils.data.Dataset):
    '''
    Data reader
    '''
    def __init__(self, root, extensions, dimension, print_ref=False):
        self.samples = self._make_dataset(root, extensions)
        self.root = root
        self.dimension = dimension
        if len(self.samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        # Normalization for VGG
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((-1, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((-1, 1, 1))
        # Print 3D model _ref
        self.print_ref = print_ref

    def __getitem__(self, index):
        path = self.samples[index]
        ims, viewpoints, points, normals = self._loader(path)

        # Apply small transform
        ims = ims.astype(float)/255.0
        ims = np.transpose(ims, (0, 3, 1, 2))
        ims = (ims - self.mean)/self.std

        return torch.from_numpy(ims).float(), \
               torch.from_numpy(viewpoints).float(), \
               torch.from_numpy(points).float(), \
               torch.from_numpy(normals).float()

    def __len__(self):
        return len(self.samples)

    def _loader(self, path):
        if self.print_ref:
            print(path)
        stl_indices = np.load(self.root + 'state_files/asteroid_choice.npy')
        stl_index = int(path[-8:-5])
        orbits_pos = np.load(self.root + 'state_files/orbits_positions.npy')
        orbits_att = np.load(self.root + 'state_files/orbits_attitudes.npy')
        img_indices = np.arange(100)
        np.random.shuffle(img_indices)
        ims = []
        viewpoints = []
        for i in range(self.dimension):
            ii = img_indices[i]
            img_path = path
            if ii < 10:
                img_path += "0"
            img_path += str(ii) + ".png"
            im = io.imread(img_path)
            im[np.where(im[:, :, 3] == 0)] = 255
            im = im[:, :, :3].astype(np.float32)
            ims.append(im)
            viewpoint = np.zeros(7)
            viewpoint[:3] = orbits_pos[stl_index, ii, :]
            viewpoint[3:] = orbits_att[stl_index, ii, :]
            viewpoints.append(viewpoint)
        stl_files = ['bennu.stl', 'itokawa.stl', 'mithra.stl', 'toutatis.stl']
        my_mesh = mesh.Mesh.from_file(self.root + 'stl_files/' + stl_files[stl_indices[stl_index]])
        normals = my_mesh.normals.astype(float)
        npy_files = ['bennu.npy', 'itokawa.npy', 'mithra.npy', 'toutatis.npy']
        point_indices = np.arange(20000)
        np.random.shuffle(point_indices)
        point_indices = point_indices[:8853]
        points = np.load(self.root + 'stl_files/' + npy_files[stl_indices[stl_index]])
        points = points[point_indices]
        return np.asarray(ims), np.asarray(viewpoints), np.asarray(points), np.asarray(normals)

    def _make_dataset(self, dir, extensions):
        paths = []
        stl_indices = np.load(dir + 'state_files/asteroid_choice.npy')
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                item_visited = ""
                for fname in sorted(fnames):
                    if self._has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = path[:-6] # strip away **.png
                        stl_index = int(item[-8:-5])
                        if item != item_visited and stl_indices[stl_index] != 2:
                            item_visited = item
                            paths.append(item)
        return paths

    def _has_file_allowed_extension(self, filename, extensions):
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)
