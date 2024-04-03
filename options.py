import argparse

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='UNet Training')
        self.initialize()

    def initialize(self):
        # Data parameters
        self.parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
        self.parser.add_argument('--no_of_workers', type=int, default=8, help='Number of workers for DataLoader')

        # Training parameters
        self.parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
        self.parser.add_argument('--accumulation_steps', type=int, default=8, help='Number of steps to accumulate gradients')

        # Device
        self.parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')

        # Model parameters
        self.parser.add_argument('--model_path', type=str, default='ckpt/unet_model_latest.pth', help='Path to save trained model')
        self.parser.add_argument('--save_every' , type = int , default = 100 , help = 'Model Checkpoints saving')

        # DataLoader parameters
        self.parser.add_argument('--train_dataset_path', type=str, default='data', help='Path to training dataset')

    def parse(self):
        return self.parser.parse_args()

options = Options().parse()
