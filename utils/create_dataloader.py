from utils.dataset import ScatteringFlareDataset
from torch.utils.data import DataLoader

# def training_dataloader():

#     dataset = ScatteringFlareDataset()

#     dataloader = DataLoader(dataset)

#     return dataset , dataloader

def training_dataloader(batch_size, num_workers):
    dataset = ScatteringFlareDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataset, dataloader
