from torch.utils.data import Dataset

class TextDataset(Dataset):
     def __init__(self, text_data: str, seq_length: int = 25) 