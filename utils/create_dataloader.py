from utils.dataset import ScatteringFlareDataset
from torch.utils.data import DataLoader

def training_dataloader(batch_size, num_workers , data_path):
    dataset = ScatteringFlareDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataset, dataloader
