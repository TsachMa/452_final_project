from data_utils import *
from models import *
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = cfg.data.data_dir
    dataObject = xrdData(data_dir, device)

    dataObject.make_datasets(cfg.data.fraction_of_data, cfg.data.composition_embedding)

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(dataObject.torch_datasets['train'],  batch_size=cfg.data.batch_size, shuffle=True)
    valid_loader = DataLoader(dataObject.torch_datasets['val'],  batch_size=cfg.data.batch_size, shuffle=False)  

    # Create the model instance and move it to the selected device
    model = XRD_C_SymNet(in_channels=cfg.model.in_channels, output_dim=cfg.model.output_dim, composition_model=cfg.model.composition_model).to(device)

    # Define optimizer and loss function
    weight_decay = 0  # Example value, adjust based on your needs
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    max_epochs = cfg.training.max_epochs
    metrics = ["accuracy", "loss"]

    log = {
        f"{type}": {f"{metric}" : np.zeros(max_epochs) for metric in metrics} for type in ['train', 'val']     
    }

    for epoch in range(max_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for xrd, composition, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(xrd, composition)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_accuracy = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        total_valid_loss = 0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():  # No gradients needed for validation
            for xrd, composition, targets in valid_loader:
                outputs = model(xrd, composition)
                loss = criterion(outputs, targets)
                total_valid_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1) 
                total_valid += targets.size(0)
                correct_valid += (predicted == targets).sum().item()

        valid_accuracy = 100 * correct_valid / total_valid

        total_train_loss = total_train_loss / len(train_loader)
        validation_loss = total_valid_loss / len(valid_loader)

        print(f"Epoch {epoch+1}, Training Loss: {total_train_loss}, Training Accuracy: {train_accuracy}%, Validation Loss: {validation_loss}, Validation Accuracy: {valid_accuracy}%")

        log['train']['accuracy'][epoch] = (train_accuracy)
        log['train']['loss'][epoch] = (total_train_loss)

        log['val']['accuracy'][epoch] = (valid_accuracy)
        log['val']['loss'][epoch] = (validation_loss)

    torch.save(model, 'model.pth')

    for data_type, metrics_dict in log.items():
        for metric, array in metrics_dict.items():
            filename = f"{data_type}_{metric}.npy"  # Construct filename, e.g., "train_accuracy.npy"
            np.save(filename, array)  # Save the array to a file

if __name__ == "__main__":
    train()