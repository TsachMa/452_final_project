
import torch
import os
from torch.utils.data import Dataset

class merged_Dataset(Dataset):
    """
    A custom dataset class that combines XRD, composition, and target tensors.

    Args:
        xrd (torch.Tensor): The XRD tensor.
        composition (torch.Tensor): The composition tensor.
        targets (torch.Tensor): The target tensor.

    Attributes:
        xrd (torch.Tensor): The XRD tensor.
        composition (torch.Tensor): The composition tensor.
        targets (torch.Tensor): The target tensor.
    """

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
    """
    A class that represents XRD data.

    Parameters:
    - data_dir (str): The directory where the data is stored.
    - device (str): The device to move the data to.
    - data_to_normalize (list): A list of data attributes to normalize.
    - datasets_avail (list): A list of available datasets.

    Attributes:
    - data_dir (str): The directory where the data is stored.
    - datasets_avail (list): A list of available datasets.
    - device (str): The device to move the data to.
    - data_attributes (dict): A dictionary to hold data attributes.
    - num_samples (int): The number of samples in the data.

    Methods:
    - get_data(suffix): Loads and returns data for the given suffix.
    - move_to_device(): Moves the data to the specified device.
    - normalize(data): Normalizes the given data.
    - make_dataset(type, composition_embedding, amt_of_data): Creates a dataset.
    - make_datasets(fraction_of_total_data, composition_embedding): Creates datasets based on the fraction of total data.
    """

    def __init__(self, data_dir, device, 
                 data_to_normalize=["compositionseq"],
                 datasets_avail=["train", "val"]):
        """
        Initializes an instance of the xrdData class.

        Parameters:
        - data_dir (str): The directory where the data is stored.
        - device (str): The device to move the data to.
        - data_to_normalize (list): A list of data attributes to normalize.
        - datasets_avail (list): A list of available datasets.
        """

        self.data_dir = data_dir
        self.datasets_avail = datasets_avail
        self.device = device
        
        # Initialize a dictionary to hold data attributes
        self.data_attributes = {}

        # Load and assign data
        self.data_attributes['xrds'] = self.get_data("pvs")
        self.data_attributes['sgs'] = self.get_data("sgs")
        self.data_attributes['compositionseq'] = self.get_data('compositionseq')

        # Move data to device
        self.move_to_device()

        # Normalize 
        for data in data_to_normalize:
            self.normalize(self.data_attributes[data])

        # Get num samples
        self.num_samples = len(self.data_attributes['xrds']['train'])

    def get_data(self, suffix):
        """
        Loads and returns data for the given suffix.

        Parameters:
        - suffix (str): The suffix of the data file.

        Returns:
        - data (dict): The loaded data.
        """
        return {type: torch.load(os.path.join(self.data_dir, f"{type}_{suffix}.pt")) for type in self.datasets_avail}

    def move_to_device(self): 
        """
        Moves the data to the specified device.
        """
        for key, value in self.data_attributes.items():
            for k, v in value.items():
                value[k] = v.to(self.device)

    def normalize(self, data): 
        """
        Normalizes the given data.

        Parameters:
        - data (dict): The data to be normalized.
        """
        mean = data["train"].mean()
        std = data["train"].std()
        for key, value in data.items():
            data[key] = (value - mean) / std

    def make_dataset(self, type, composition_embedding, amt_of_data):
        """
        Creates a dataset.

        Parameters:
        - type (str): The type of dataset.
        - composition_embedding (str): The composition embedding.
        - amt_of_data (int): The amount of data to include in the dataset.

        Returns:
        - dataset (merged_Dataset): The created dataset.
        """
        return merged_Dataset(self.data_attributes['xrds'][type][:amt_of_data], 
                                              self.data_attributes[composition_embedding][type][:amt_of_data], 
                                              self.data_attributes['sgs'][type][:amt_of_data])

    def make_datasets(self, fraction_of_total_data, composition_embedding):
        """
        Creates datasets based on the fraction of total data.

        Parameters:
        - fraction_of_total_data (float): The fraction of total data to include in the training dataset.
        - composition_embedding (str): The composition embedding.

        Returns:
        - torch_datasets (dict): A dictionary of created datasets.
        """
        amt_of_training_data = int(fraction_of_total_data * self.num_samples) #only fractionate the training data 

        data_and_amt = zip(self.datasets_avail, [amt_of_training_data, 1*len(self.data_attributes['xrds']['val']), 1*len(self.data_attributes['xrds']['val'])]) # use full val data

        #make pytorch datasets
        self.composition_embedding = composition_embedding
        self.torch_datasets = {type: self.make_dataset(type, composition_embedding, amt_of_data) for type, amt_of_data in data_and_amt}

class ExperimentalSimulation():
    """
    A class that performs experimental simulation on XRD data.

    Args:
        device (torch.device): The device to perform the simulation on.
        crop_start (int, optional): The starting point for cropping the XRD data. Defaults to 4000.
        crop_stop (int, optional): The stopping point for cropping the XRD data. Defaults to 4000.
        noise_range (float, optional): The range of noise to add to the XRD data. Defaults to 0.4.
        drop_width (int, optional): The width of the dropped regions in the XRD data. Defaults to 1000.
        drop_freq (int, optional): The frequency of dropped regions in the XRD data. Defaults to 2.

    Attributes:
        crop_range (list): A list containing the crop start and stop indices.
        noise_range (float): The range of noise to add to the XRD data.
        drop_width (int): The width of the dropped regions in the XRD data.
        drop_freq (int): The frequency of dropped regions in the XRD data.
        device (torch.device): The device to perform the simulation on.
    """

    def __init__(self, device, crop_start=4000, crop_stop=4000, noise_range=0.4, drop_width=1000, drop_freq=2):
        self.crop_range = [crop_start, crop_stop]
        self.noise_range = noise_range
        self.drop_width = drop_width
        self.drop_freq = drop_freq
        self.device = device

    def random_crop(self, xrd):
        """
        Randomly crops the XRD data.

        Args:
            xrd (torch.Tensor): The input XRD data.

        Returns:
            torch.Tensor: The cropped XRD data.
        """
        n_rows, n_cols = xrd.shape[0], xrd.shape[2]

        start_indices = torch.randint(0, self.crop_range[0], (n_rows, 1))
        end_indices = torch.randint(n_cols - self.crop_range[1], n_cols, (n_rows, 1))

        range_tensor = torch.arange(n_cols).unsqueeze(0).expand(n_rows, -1)
        mask = (range_tensor >= start_indices) & (range_tensor < end_indices)

        return mask.unsqueeze(1).to(self.device) * xrd

    def random_noise(self, xrd):
        """
        Adds random noise to the XRD data.

        Args:
            xrd (torch.Tensor): The input XRD data.

        Returns:
            torch.Tensor: The XRD data with added noise.
        """
        n_rows, n_cols = xrd.shape[0], xrd.shape[2]
        noise = torch.rand(n_rows, 1, n_cols) * self.noise_range

        return xrd + noise.to(self.device)

    def random_drops(self, xrd):
        """
        Adds random dropped regions to the XRD data.

        Args:
            xrd (torch.Tensor): The input XRD data.

        Returns:
            torch.Tensor: The XRD data with added dropped regions.
        """
        n_rows, n_cols = xrd.shape[0], xrd.shape[2]

        drop_indices = torch.randint(0, n_cols, (n_rows, self.drop_freq))

        mask = torch.zeros((n_rows, n_cols))
        range_tensor = torch.arange(n_cols).unsqueeze(0).expand(n_rows, -1)

        for i in range(self.drop_freq):
            mask += (range_tensor >= drop_indices[:, i].unsqueeze(1) + self.drop_width)
            mask += (range_tensor < drop_indices[:, i].unsqueeze(1) - self.drop_width)

        mask[torch.where(mask != self.drop_freq)] = 0
        mask /= self.drop_freq

        return mask.unsqueeze(1).to(self.device) * xrd

    def sim(self, xrd):
        """
        Performs the experimental simulation on the XRD data.

        Args:
            xrd (torch.Tensor): The input XRD data.

        Returns:
            torch.Tensor: The simulated XRD data.
        """
        xrd = self.random_crop(xrd)
        xrd = self.random_drops(xrd)
        xrd = self.random_noise(xrd)

        return xrd
    
def tokenize_xrd(xrd, token_size, seq_len=8500):
    """
    Tokenizes the XRD data by reshaping the input array.

    Args:
        xrd (ndarray): The input XRD data array of shape (n, m, seq_length), where n is the number of samples,
                       m is the number of features, and seq_length is the length of the original sequence.
        token_size (int): The size of each token.
        seq_len (int, optional): The desired sequence length after tokenization. Defaults to 8500.

    Returns:
        ndarray: The tokenized XRD data array of shape (n, seq_len/token_size, token_size), where n is the number of samples,
                 seq_len/token_size is the number of tokens, and token_size is the size of each token.
    """

    # Get the number of samples (n) and the length of the original sequence
    n, _, seq_length = xrd.shape
    
    # Reshape the vector
    output_vector = xrd.reshape(n, int(seq_len / token_size), token_size)
    
    return output_vector

def untokenize_xrd(tokenized_xrd, token_size, seq_len=8500):
    """
    Reshapes a tokenized XRD sequence back to its original shape.

    Args:
        tokenized_xrd (ndarray): A 3-dimensional numpy array representing the tokenized XRD sequence.
        token_size (int): The size of each token.
        seq_len (int, optional): The length of the original XRD sequence. Defaults to 8500.

    Returns:
        ndarray: A 3-dimensional numpy array representing the reshaped original XRD sequence.

    Raises:
        ValueError: If the tokenized sequence length and token size do not match the expected original sequence length.

    """

    # Get the number of samples (n) and the length of the tokenized sequence
    n, seq_token_length, _ = tokenized_xrd.shape
    
    # Validate that the number of tokens times the token size equals the original sequence length
    if seq_token_length * token_size != seq_len:
        raise ValueError(f"Tokenized sequence length and token size do not match the expected original sequence length of {seq_len}")
    
    # Reshape the vector back to the original shape
    original_vector = tokenized_xrd.reshape(n, 1, seq_len)
    
    return original_vector
