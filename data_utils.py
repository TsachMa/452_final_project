
import torch
import os
from torch.utils.data import Dataset

class merged_Dataset(Dataset):
    def __init__(self, xrd, composition, targets):
        assert xrd.size(0) == composition.size(0) == targets.size(0), "The number of elements in both tensor sets should be the same"
        self.xrd = xrd
        self.composition = composition
        self.targets = targets

    def __len__(self):
        return self.xrd.size(0)

    def __getitem__(self, index):
        return (self.xrd[index], self.composition[index], self.targets[index])
    
class xrdData():
    def __init__(self, data_dir, device, 
                 data_to_normalize = ["composition1D", "composition2D"],
                 datasets_avail = ["train", "val"]):
        
        self.data_dir = data_dir
        self.datasets_avail = datasets_avail
        self.device = device
        
        # Initialize a dictionary to hold data attributes
        self.data_attributes = {}

        # Load and assign data
        self.data_attributes['xrds'] = self.get_data("pvs")
        self.data_attributes['sgs'] = self.get_data("sgs")
        self.data_attributes['composition1D'] = self.get_data("composition")
        self.data_attributes['composition2D'] = self.get_data("composition2D")

        # Move data to device
        self.move_to_device()

        # Normalize 
        for data in data_to_normalize:
            self.normalize(self.data_attributes[data])

        #Get num samples
        self.num_samples = len(self.data_attributes['xrds']['train'])

    def get_data(self, suffix):
        return {type: torch.load(os.path.join(self.data_dir, f"{type}_{suffix}.pt")) for type in self.datasets_avail}

    def move_to_device(self): 
        for key, value in self.data_attributes.items():
            for k, v in value.items():
                value[k] = v.to(self.device)

    def normalize(self, data): 
        mean = data["train"].mean()
        std = data["train"].std()
        for key, value in data.items():
            data[key] = (value - mean) / std

    def make_dataset(self, type, composition_embedding, amt_of_data):
        return merged_Dataset(self.data_attributes['xrds'][type][:amt_of_data], 
                                              self.data_attributes[composition_embedding][type][:amt_of_data], 
                                              self.data_attributes['sgs'][type][:amt_of_data])

    def make_datasets(self, fraction_of_total_data, composition_embedding):

        amt_of_training_data = int(fraction_of_total_data * self.num_samples) #only fractionate the training data 

        data_and_amt = zip(self.datasets_avail, [amt_of_training_data, 1*len(self.data_attributes['xrds']['val'])]) # use full val data

        #make pytorch datasets
        self.composition_embedding = composition_embedding
        self.torch_datasets = {type: self.make_dataset(type, composition_embedding, amt_of_data) for type, amt_of_data in data_and_amt} 

class ExperimentalSimulation(): 
    def __init__(self, device, crop_start = 4000,
                        crop_stop = 4000,
                        noise_range = 0.4,
                        drop_width = 1000,
                        drop_freq = 2,
                        ):
        
        self.crop_range = [crop_start, crop_stop]
        self.noise_range = noise_range
        self.drop_width = drop_width
        self.drop_freq = drop_freq
        self.device = device

    def random_crop(self, xrd):
        n_rows, n_cols = xrd.shape[0], xrd.shape[2]

        # Generate random start and end indices for each row
        start_indices = torch.randint(0, self.crop_range[0], (n_rows, 1))
        end_indices = torch.randint(n_cols - self.crop_range[1], n_cols, (n_rows, 1))

        # Create a range tensor
        range_tensor = torch.arange(n_cols).unsqueeze(0).expand(n_rows, -1)
        # Create the mask by comparing the range tensor with start and end indices
        mask = (range_tensor >= start_indices) & (range_tensor < end_indices)

        return mask.unsqueeze(1).to(self.device) * xrd

    def random_noise(self, xrd):
        n_rows, n_cols = xrd.shape[0], xrd.shape[2]
        noise = torch.rand(n_rows, 1, n_cols) * self.noise_range

        return xrd + noise.to(self.device)

    def random_drops(self, xrd):
        n_rows, n_cols = xrd.shape[0], xrd.shape[2]

        drop_indices = torch.randint(0, n_cols, (n_rows, self.drop_freq))

        mask = torch.zeros((n_rows, n_cols))

        # Create a range tensor
        range_tensor = torch.arange(n_cols).unsqueeze(0).expand(n_rows, -1)

        for i in range(self.drop_freq):
            # Create the mask by comparing the range tensor with start and end indices
            mask += (range_tensor >= drop_indices[:, i].unsqueeze(1) + self.drop_width) 
            mask += (range_tensor < drop_indices[:, i].unsqueeze(1)  - self.drop_width)

        mask[torch.where(mask != self.drop_freq)] = 0
        mask /= self.drop_freq

        return mask.unsqueeze(1).to(self.device) * xrd
    
    def sim(self, xrd):
        xrd = self.random_crop(xrd)
        xrd = self.random_noise(xrd)
        xrd = self.random_drops(xrd)

        return xrd