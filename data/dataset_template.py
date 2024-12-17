from torch.utils.data import Dataset
import cv2

class MyData(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: Input data (e.g., numpy array, pandas DataFrame, or torch.Tensor).
            labels: Corresponding labels for the data.
            transform: Optional; a function/transform to apply to the data.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple (sample, label). If a transform is provided, it is applied to the sample.
        """
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

    def _other_function(self, x):
        """
        (Optional) Define helper functions to be used within __getitem__.
        """
        # Example: Normalize data
        return x / 255.0
