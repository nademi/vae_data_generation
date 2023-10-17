import torch
from torch.utils.data import Dataset

class MockECGDataLoader(Dataset):
    def __init__(self, num_samples=1000, ecg_length=4096, num_leads=12):
        """
        A mock dataloader to simulate 12-lead ECG data.
        :param num_samples: Number of ECG samples in this dataset.
        :param ecg_length: Number of data points in one ECG sample.
        :param num_leads: Number of ECG leads.
        """
        self.num_samples = num_samples
        self.ecg_length = ecg_length
        self.num_leads = num_leads
        self.data = torch.randn(num_samples, num_leads, ecg_length)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

if __name__ == "__main__":
    # Testing the mock dataloader
    dataset = MockECGDataLoader()
    print(dataset[0].shape)  # Expected output: torch.Size([12, 4096])
