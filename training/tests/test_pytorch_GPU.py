import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# built-in datasets:
from torchvision import datasets, transforms
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


class IrisClassifier(nn.Module):
    """
    A simple feed-forward neural network for multi-class classification,
    designed for a dataset like the Iris dataset (4 features, 3 classes).
    """
    def __init__(self, input_size, num_classes):
        """
        Initializes the IrisClassifier model.

        Args:
            input_size (int): The number of input features per sample.
                              For Iris, this is typically 4.
            num_classes (int): The number of output classes.
                               For Iris, this is typically 3.
        """
        super(IrisClassifier, self).__init__()
        # First linear layer: input_size features to a hidden layer of 64 neurons
        self.fc1 = nn.Linear(input_size, 64)
        # Second linear layer: 64 hidden neurons to another hidden layer of 32 neurons
        self.fc2 = nn.Linear(64, 32)
        # Output linear layer: 32 hidden neurons to num_classes output logits
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of features.
                              Expected shape: (batch_size, input_size)

        Returns:
            torch.Tensor: The output logits for each class.
                          Expected shape: (batch_size, num_classes)
        """
        # Flatten the input tensor from (batch_size, C, H, W) to (batch_size, C*H*W)
        # The -1 means infer the batch size, and the rest of the dimensions are flattened.
        x = x.view(x.size(0), -1) # x.size(0) gets the batch size
        
        # Apply first linear layer and ReLU activation
        x = F.relu(self.fc1(x))
        # Apply second linear layer and ReLU activation
        x = F.relu(self.fc2(x))
        # Apply output linear layer
        x = self.fc3(x)
        # Note: We typically don't apply Softmax here if using nn.CrossEntropyLoss,
        # as CrossEntropyLoss includes Softmax internally for numerical stability.
        return x


if __name__ == '__main__':
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")

    # If CUDA is available, print the number of GPUs
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        # Print the name of the current GPU
        print(f"Current GPU name: {torch.cuda.get_device_name(0)}") # 0 refers to the first GPU

    ##################################################
    # Define the Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model = MyModel()

    ##################################################
    # Move the Model to the GPU
    model.to(device) # <------------------------------------------ Move model to the GPU
    print(model)


    ##################################################
    # Move input data and target label  (Tensors) to the GPU

    print("##################################################")
    print("# Test 1 single forward pass")
    # Create a dummy input tensor on the CPU
    input_tensor = torch.randn(64, 10) # Batch size 64, 10 features

    # Move the input tensor to the GPU
    input_tensor = input_tensor.to(device) # <------------------- Move input data to the GPU

    # Example of a single  forward pass
    output = model(input_tensor)
    print(output.shape)
    print(output.device) # Should show 'cuda:0'





    ##################################################
    # Batches: When data is loaded data during training, ensure each batch is moved to the GPU:

    print("##################################################")
    print("# Test 2 batches")

    # Define a simple transform for FashionMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),           # Converts a PIL Image or NumPy array to a PyTorch Tensor (HWC to CWH, scales to [0, 1])
        transforms.Normalize((0.5,), (0.5,))  # Normalizes the tensor with mean 0.5 and std 0.5
    ])

    # Load a built-in dataset (e.g., FashionMNIST)
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,       # Number of samples per batch
        shuffle=True,        # Shuffle data at the beginning of each epoch (usually for training)
        num_workers=4,       # Number of subprocesses to use for data loading
        pin_memory=True      # Optimize data transfer to GPU
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,       # No need to shuffle test data
        num_workers=4,
        pin_memory=True
    )

    input_feature_size = 28*28
    number_of_output_classes = 10
    # Create an instance of the model
    model = IrisClassifier(input_size=input_feature_size, num_classes=number_of_output_classes)  
     # Print the model architecture
    print("--- IrisClassifier Model Architecture ---")
    print(model)

    # --- Device Configuration ---
    # Determine which device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Move the entire model to the selected device
    model.to(device) # <-------------------------------------------- Move model to the GPU
    print(f"Model moved to device: {next(model.parameters()).device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Adam Optimizer initialized with learning rate: {optimizer.defaults['lr']}")
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Move data and labels to the appropriate device (GPU if available)
            data = data.to(device) # <-------------------------- Move input data to the GPU
            labels = labels.to(device) # <---------------------- Move input data to the GPU

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass (compute gradients)
            loss.backward()

            # Update model parameters
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # (Optional) Evaluate on the test set after each epoch
        model.eval() # Set model to evaluation mode
        test_loss = 0
        correct = 0
        with torch.no_grad(): # Disable gradient calculation for inference
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_dataset)
        print(f'Epoch {epoch+1} Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')