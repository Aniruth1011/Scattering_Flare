import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.create_dataloader import training_dataloader
from net.model import UNet
from tqdm import tqdm 

# Define hyperparameters
batch_size = 16
num_epochs = 10
learning_rate = 0.001


train_dataset , train_dataloader = training_dataloader()

model = UNet()
device = "cuda"

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    
    for flary_img , deflared_img in train_dataloader:

        outputs = model(flary_img)
        loss = criterion(outputs, deflared_img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() 
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), 'ckpt/unet_model.pth')
