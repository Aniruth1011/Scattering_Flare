# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from utils.create_dataloader import training_dataloader
# from net.model import UNet
# from tqdm import tqdm 

# batch_size = 8
# num_epochs = 1
# learning_rate = 0.001
# no_of_workers = 8

# train_dataset , train_dataloader = training_dataloader(batch_size, no_of_workers)

# device = "cuda"
# model = UNet().to(device)

# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# if __name__ == '__main__':
#     # for epoch in tqdm(range(num_epochs)):
#     #     model.train().cuda()
#     #     running_loss = 0.0
        
#     #     for flary_img , deflared_img in train_dataloader:
#     #         flary_img = flary_img.to(device)
#     #         deflared_img = deflared_img.to(device)
            
#     #         outputs = model(flary_img)
#     #         loss = criterion(outputs, deflared_img)
            
#     #         optimizer.zero_grad()
#     #         loss.backward()
#     #         optimizer.step()
#     #         running_loss += loss.item() 
#     #         print('loss: ', running_loss)

#     accumulation_steps = 8
#     running_loss = 0.0
#     for epoch in tqdm(range(num_epochs)):
#         model.train().cuda()
        
#         for batch_idx, (flary_img, deflared_img) in enumerate(train_dataloader):
#             flary_img = flary_img.to(device)
#             deflared_img = deflared_img.to(device)
            
#             outputs = model(flary_img)
#             loss = criterion(outputs, deflared_img)
#             loss = loss / accumulation_steps  # Divide loss by accumulation steps
            
#             loss.backward()
            
#             if (batch_idx + 1) % accumulation_steps == 0:  # Update weights every accumulation_steps
#                 optimizer.step()
#                 optimizer.zero_grad()
            
#             running_loss += loss.item()
#             print(running_loss)
        
#         epoch_loss = running_loss / len(train_dataset)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

#     torch.save(model.state_dict(), 'ckpt/unet_model.pth')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.create_dataloader import training_dataloader
from net.model import UNet
from tqdm import tqdm 

batch_size = 8
num_epochs = 10
learning_rate = 0.001
no_of_workers = 8

train_dataset , train_dataloader = training_dataloader(batch_size, no_of_workers)

device = "cuda"
model = UNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load previously saved model weights if available
# weight_file = 'ckpt/unet_model.pth'
# if torch.cuda.is_available():
#     model.load_state_dict(torch.load(weight_file))
#     print("Model loaded successfully to gpu.")
# else:
#     model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
#     print("Model loaded successfully to cpu.")

# Resume training
if __name__ == '__main__':
    accumulation_steps = 8
    
    for epoch in tqdm(range(num_epochs)):
        model.train().cuda()
        running_loss = 0.0
        
        for batch_idx, (flary_img, deflared_img) in enumerate(train_dataloader):
            flary_img = flary_img.to(device)
            deflared_img = deflared_img.to(device)
            
            outputs = model(flary_img)
            loss = criterion(outputs, deflared_img)
            loss = loss / accumulation_steps  # Divide loss by accumulation steps
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:  # Update weights every accumulation_steps
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item()
            loss_copy = running_loss
    
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        print(loss_copy)
        print(len(train_dataset))
        torch.save(model.state_dict(), f'ckpt/unet_model_{epoch}.pth')

    torch.save(model.state_dict(), 'ckpt/unet_model_latest.pth')