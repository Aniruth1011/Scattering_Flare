from utils.dataset import ScatteringFlareDataset
from torch.utils.data import  DataLoader

def training_dataloader():

    dataset = ScatteringFlareDataset()

    dataloader = DataLoader(dataset)

    return dataset , dataloader
