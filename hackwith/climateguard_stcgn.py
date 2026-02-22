"""
ClimateGuard - Spatio-Temporal Graph Convolutional Network (STGCN) Implementation
This module implements the core STGCN model for climate risk prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import numpy as np


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network component
    Captures sequential patterns in time-series data
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(out_channels, out_channels, (1, kernel_size))
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, features, nodes, time_steps)
        Returns:
            Tensor of shape (batch, out_channels, nodes, time_steps - kernel_size + 1)
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        return F.relu(x)


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block
    Combines graph convolution (spatial) with temporal convolution
    """
    def __init__(self, in_channels, spatial_channels, out_channels, 
                 num_nodes, K=3, kernel_size=3):
        super(STConvBlock, self).__init__()
        
        # Temporal convolution at input
        self.temporal1 = TemporalConvNet(in_channels, in_channels, kernel_size)
        
        # Graph convolution for spatial dependencies
        self.graph_conv = ChebConv(in_channels, spatial_channels, K)
        
        # Temporal convolution at output
        self.temporal2 = TemporalConvNet(spatial_channels, out_channels, kernel_size)
        
        # Residual connection
        self.residual_conv = nn.Conv2d(in_channels, out_channels, (1, 1)) \
                             if in_channels != out_channels else None
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Input features (batch, features, nodes, time_steps)
            edge_index: Graph edge connections
            edge_weight: Optional edge weights
        Returns:
            Output features (batch, out_channels, nodes, new_time_steps)
        """
        batch_size, in_channels, num_nodes, time_steps = x.shape
        
        # Temporal convolution
        residual = x
        x = self.temporal1(x)
        
        # Reshape for graph convolution: (batch * time_steps, nodes, features)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size * x.size(1), num_nodes, -1)
        
        # Graph convolution
        x = self.graph_conv(x, edge_index, edge_weight)
        x = F.relu(x)
        
        # Reshape back: (batch, features, nodes, time_steps)
        x = x.view(batch_size, -1, num_nodes, x.size(-1))
        x = x.permute(0, 3, 2, 1).contiguous()
        
        # Temporal convolution
        x = self.temporal2(x)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        # Match temporal dimensions
        if residual.size(-1) != x.size(-1):
            residual = residual[..., :x.size(-1)]
        
        x = x + residual
        x = self.bn(x)
        return F.relu(x)


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for Climate Prediction
    
    Architecture:
        - Multiple ST-Conv blocks to capture spatial and temporal patterns
        - Temporal attention for important time steps
        - Fully connected layers for final prediction
    """
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, num_classes=3):
        """
        Args:
            num_nodes: Number of geographic regions/nodes
            num_features: Number of input features per node
            num_timesteps_input: Number of historical time steps
            num_timesteps_output: Number of future time steps to predict
            num_classes: Number of risk classes (default: 3 for Safe/Warning/Danger)
        """
        super(STGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_timesteps_output = num_timesteps_output
        
        # ST-Conv Block 1
        self.block1 = STConvBlock(
            in_channels=num_features,
            spatial_channels=64,
            out_channels=64,
            num_nodes=num_nodes,
            K=3,
            kernel_size=3
        )
        
        # ST-Conv Block 2
        self.block2 = STConvBlock(
            in_channels=64,
            spatial_channels=128,
            out_channels=128,
            num_nodes=num_nodes,
            K=3,
            kernel_size=3
        )
        
        # ST-Conv Block 3
        self.block3 = STConvBlock(
            in_channels=128,
            spatial_channels=128,
            out_channels=128,
            num_nodes=num_nodes,
            K=3,
            kernel_size=3
        )
        
        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Output layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_timesteps_output * num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass
        
        Args:
            x: Input features (batch, features, nodes, time_steps)
            edge_index: Graph edge connections
            edge_weight: Optional edge weights
            
        Returns:
            predictions: Risk predictions (batch, nodes, time_steps, classes)
        """
        # ST-Conv blocks
        x = self.block1(x, edge_index, edge_weight)
        x = self.block2(x, edge_index, edge_weight)
        x = self.block3(x, edge_index, edge_weight)
        
        # Temporal attention
        # x shape: (batch, channels, nodes, time_steps)
        batch_size, channels, num_nodes, time_steps = x.shape
        
        # Reshape for attention: (batch * nodes, time_steps, channels)
        x_att = x.permute(0, 2, 3, 1).contiguous()
        x_att = x_att.view(batch_size * num_nodes, time_steps, channels)
        
        # Compute attention weights
        att_weights = self.attention(x_att)  # (batch * nodes, time_steps, 1)
        att_weights = F.softmax(att_weights, dim=1)
        
        # Apply attention
        x_att = x_att * att_weights
        x_att = torch.sum(x_att, dim=1)  # (batch * nodes, channels)
        
        # Fully connected layers
        x_att = self.dropout(x_att)
        x_att = F.relu(self.fc1(x_att))
        x_att = self.dropout(x_att)
        x_att = self.fc2(x_att)
        
        # Reshape to output format: (batch, nodes, time_steps, classes)
        predictions = x_att.view(batch_size, num_nodes, 
                                self.num_timesteps_output, -1)
        
        return predictions


class ClimateRiskPredictor:
    """
    High-level wrapper for climate risk prediction
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def predict_flood_risk(self, features, edge_index, edge_weight=None):
        """
        Predict flood risk for regions
        
        Args:
            features: Climate features (batch, features, nodes, time_steps)
            edge_index: Graph structure
            edge_weight: Optional edge weights
            
        Returns:
            flood_probabilities: Probability of flooding (batch, nodes, time_steps)
            risk_levels: Risk classification (batch, nodes, time_steps)
        """
        with torch.no_grad():
            features = features.to(self.device)
            edge_index = edge_index.to(self.device)
            if edge_weight is not None:
                edge_weight = edge_weight.to(self.device)
            
            # Get predictions
            predictions = self.model(features, edge_index, edge_weight)
            
            # Convert to probabilities
            probabilities = F.softmax(predictions, dim=-1)
            
            # Extract flood risk (assuming class 2 is "Danger")
            flood_prob = probabilities[..., 2]
            
            # Classify risk levels
            risk_levels = torch.argmax(probabilities, dim=-1)
            
            return flood_prob.cpu().numpy(), risk_levels.cpu().numpy()
    
    def predict_heatwave_risk(self, features, edge_index, edge_weight=None):
        """
        Predict heatwave risk for regions
        Similar to flood risk but may use different features
        """
        return self.predict_flood_risk(features, edge_index, edge_weight)


def build_spatial_graph(locations, threshold_distance=50.0):
    """
    Build spatial graph from geographic locations
    
    Args:
        locations: Array of (latitude, longitude) coordinates
        threshold_distance: Maximum distance (km) for edge connection
        
    Returns:
        edge_index: Edge connectivity tensor
        edge_weight: Edge weight tensor (inverse distance)
    """
    from scipy.spatial.distance import cdist
    
    # Calculate pairwise distances (using haversine would be more accurate)
    distances = cdist(locations, locations, metric='euclidean')
    
    # Create edges for nearby locations
    edges = []
    weights = []
    
    num_nodes = len(locations)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if distances[i, j] < threshold_distance:
                edges.append([i, j])
                edges.append([j, i])  # Undirected graph
                weight = 1.0 / (distances[i, j] + 1e-6)  # Inverse distance
                weights.append(weight)
                weights.append(weight)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    return edge_index, edge_weight


def train_model(model, train_loader, val_loader, num_epochs=100, 
                learning_rate=0.001, device='cuda'):
    """
    Training loop for STGCN model
    
    Args:
        model: STGCN model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Training device
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Weighted loss for imbalanced classes
    class_weights = torch.tensor([1.0, 2.0, 5.0], device=device)  
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            features, edge_index, edge_weight, labels = batch
            features = features.to(device)
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device) if edge_weight is not None else None
            labels = labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(features, edge_index, edge_weight)
            
            # Reshape for loss calculation
            batch_size, nodes, time_steps, classes = predictions.shape
            predictions = predictions.view(-1, classes)
            labels = labels.view(-1)
            
            loss = criterion(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features, edge_index, edge_weight, labels = batch
                features = features.to(device)
                edge_index = edge_index.to(device)
                edge_weight = edge_weight.to(device) if edge_weight is not None else None
                labels = labels.to(device)
                
                predictions = model(features, edge_index, edge_weight)
                
                batch_size, nodes, time_steps, classes = predictions.shape
                predictions = predictions.view(-1, classes)
                labels_flat = labels.view(-1)
                
                loss = criterion(predictions, labels_flat)
                val_loss += loss.item()
                
                predicted_classes = torch.argmax(predictions, dim=-1)
                correct += (predicted_classes == labels_flat).sum().item()
                total += labels_flat.size(0)
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Accuracy: {accuracy:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model


# Example usage
if __name__ == "__main__":
    # Model hyperparameters
    num_nodes = 50  # 50 geographic regions
    num_features = 7  # Temperature, rainfall, humidity, AQI, etc.
    num_timesteps_input = 14  # 14 days of historical data
    num_timesteps_output = 7  # Predict next 7 days
    num_classes = 3  # Safe, Warning, Danger
    
    # Initialize model
    model = STGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output,
        num_classes=num_classes
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Example input
    batch_size = 4
    x = torch.randn(batch_size, num_features, num_nodes, num_timesteps_input)
    
    # Build example graph
    locations = np.random.randn(num_nodes, 2) * 10  # Random locations
    edge_index, edge_weight = build_spatial_graph(locations, threshold_distance=15.0)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(x, edge_index, edge_weight)
    
    print(f"Output shape: {predictions.shape}")  # (batch, nodes, time_steps, classes)
    
    # Initialize predictor
    predictor = ClimateRiskPredictor(model)
    flood_prob, risk_levels = predictor.predict_flood_risk(x, edge_index, edge_weight)
    
    print(f"Flood probabilities shape: {flood_prob.shape}")
    print(f"Risk levels shape: {risk_levels.shape}")