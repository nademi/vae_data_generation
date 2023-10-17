import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.mock_dataloader import MockECGDataLoader
from models.vae import VAEModel

def kl_divergent_loss(enc_mu, enc_sd):
    return -0.5 * torch.mean(torch.sum(1 + enc_sd.pow(2).log() - enc_mu.pow(2) - enc_sd.pow(2), dim=1))

def train_vae(device, model, train_loader, val_loader, beta=8, num_epochs=3):
    mse_loss = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-8)
    model = model.to(device)
    model.train()
    min_val_loss = 40
    early_stop_count, early_stop_epoch = 0, 9
    
    for i in range(num_epochs):
        if early_stop_count >= early_stop_epoch:
            break
        train_loss = 0
        
        with tqdm(total=len(train_loader)) as pbar:
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                
                recon_x, _, (enc_mu, enc_sd) = outputs
                
                reconstruction_loss = mse_loss(recon_x, data)
                kl_loss = kl_divergent_loss(enc_mu, enc_sd)
                loss = reconstruction_loss + beta * kl_loss
                
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                pbar.set_description(f"[epoch {i+1}/{num_epochs} ]")
                pbar.set_postfix_str(f"loss = {loss.item():.4f}")
                pbar.update(1)
                
        val_loss = validate(device, model, val_loader, beta=beta)
        print (f"train_loss = {train_loss/len(train_loader):.4f} val loss = {val_loss/len(val_loader):.4f}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model, f'./torchmodel/torch_vae_12_leads_epoch_{i}.model')
            early_stop_count = 0
        else:
            early_stop_count += 1

    model.eval()
    return model

def validate(device, model, val_loader, beta=8):
    mse_loss = torch.nn.MSELoss(reduction='sum')
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for data in val_loader:
                data = data.to(device)
                outputs = model(data)
                
                recon_x, _, (enc_mu, enc_sd) = outputs
                
                reconstruction_loss = mse_loss(recon_x, data)
                kl_loss = kl_divergent_loss(enc_mu, enc_sd)
                loss = reconstruction_loss + beta * kl_loss
                
                val_loss += loss.item()
                pbar.set_postfix_str(f"loss = {loss.item():.4f}")
                pbar.update(1)
                
    return val_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAEModel()
    
    train_dataset = MockECGDataLoader()
    val_dataset = MockECGDataLoader()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    trained_model = train_vae(device, model, train_loader, val_loader)
    torch.save(trained_model, f'./outputs/torch_vae_12_leads.model')
