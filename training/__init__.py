import torch

class MockECGDataLoader(torch.utils.data.Dataset):
    def __init__(self):
        # Initialization logic for your mock dataset
        pass

    def __len__(self):
        # Return the total number of data samples
        return 100  # Placeholder value

    def __getitem__(self, idx):
        # Return a single data sample (and maybe its label)
        mock_data = torch.randn(4096, 12)  # Placeholder for mock ECG data
        return mock_data
