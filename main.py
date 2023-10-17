import torch
from datasets.mock_dataloader import MockECGDataLoader
from models.model import VAE
from training.train_vae import train_vae
from util import load_config  # Importing the load_config function from util.py

def main():
    # Load configuration
    config_path = "config.yaml"
    config = load_config(config_path)  # Using the load_config from util.py

    # Hyperparameters & Configurations
    num_samples = 1000
    ecg_length = 4096
    num_leads = 12
    batch_size = 32
    num_epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets & DataLoaders
    train_dataset = MockECGDataLoader(num_samples=num_samples)
    val_dataset = MockECGDataLoader(num_samples=int(0.2*num_samples))  # 20% for validation as an example

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = VAE(encoder_param=config["encoder_param"], decoder_param=config["decoder_param"])

    # Train
    trained_model = train_vae(device, model, train_loader, val_loader, num_epochs=num_epochs)

    # Save Model
    torch.save(trained_model.state_dict(), './outputs/torch_vae_12_leads_final.model')

if __name__ == "__main__":
    main()
