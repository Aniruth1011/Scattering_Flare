import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.create_dataloader import training_dataloader
from net.model import UNet
from tqdm import tqdm
from options import options



# DataLoader
train_dataset , train_dataloader = training_dataloader(options.batch_size, options.no_of_workers, options.train_dataset_path , options.image_size)

# Model
model = UNet().to(options.device)

# Criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=options.learning_rate)

# Resume training
if __name__ == '__main__':
    for epoch in tqdm(range(options.num_epochs)):
        model.train().to(options.device)
        running_loss = 0.0

        for batch_idx, (flary_img, deflared_img) in enumerate(train_dataloader):
            flary_img = flary_img.to(options.device)
            deflared_img = deflared_img.to(options.device)

            outputs = model(flary_img)
            loss = criterion(outputs, deflared_img)
            loss = loss / options.accumulation_steps  # Divide loss by accumulation steps

            loss.backward()

            if (batch_idx + 1) % options.accumulation_steps == 0:  # Update weights every accumulation_steps
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{options.num_epochs}], Loss: {epoch_loss:.4f}")

        if (epoch%options.save_every == 0):
            torch.save(model.state_dict(), f'ckpt/unet_model_{epoch}.pth')

    torch.save(model.state_dict(), options.model_path)
