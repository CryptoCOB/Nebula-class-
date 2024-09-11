import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
from collections import deque
from typing import Dict, Optional, List, Any
import torch.nn as nn
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralFusionModel(nn.Module):
    def __init__(self, config):
        super(NeuralFusionModel, self).__init__()
        self.config = config
        self.input_dims = config['input_dims']
        self.fusion_dim = config['fusion_dim']
        self.min_modalities = config.get('min_modalities', 2)
        self.max_modalities = config.get('max_modalities', 3)
        
        # Create adaptive layers for each possible input modality
        self.input_layers = nn.ModuleDict({
            modality: nn.Linear(dim, self.fusion_dim)
            for modality, dim in self.input_dims.items()
        })
        
        # Fusion layers for different numbers of modalities
        self.fusion_layers = nn.ModuleDict({
            str(i): nn.Linear(self.fusion_dim * i, self.fusion_dim)
            for i in range(self.min_modalities, self.max_modalities + 1)
        })
        
        self.activation = nn.ReLU()

    def forward(self, inputs):
        processed_inputs = []
        
        # Process and pad/trim inputs
        for modality, tensor in inputs.items():
            if modality in self.input_layers:
                processed = self.input_layers[modality](tensor)
                processed_inputs.append(processed)
        
        # Pad or trim to ensure we have between min_modalities and max_modalities
        num_inputs = len(processed_inputs)
        if num_inputs < self.min_modalities:
            padding = [torch.zeros_like(processed_inputs[0]) for _ in range(self.min_modalities - num_inputs)]
            processed_inputs.extend(padding)
        elif num_inputs > self.max_modalities:
            processed_inputs = processed_inputs[:self.max_modalities]
        
        # Concatenate processed inputs
        concat_inputs = torch.cat(processed_inputs, dim=-1)
        
        # Apply appropriate fusion layer
        fused = self.fusion_layers[str(len(processed_inputs))](concat_inputs)
        
        # Apply activation
        output = self.activation(fused)
        
        return output



class HolisticPerception:
    def __init__(self, config: Dict[str, Any]):
        # Ensure the config is a dictionary
        if not isinstance(config, dict):
            raise TypeError(f"Expected 'config' to be a dictionary, but got {type(config)}")
        
        self.config = config
        self.config = config
        self.input_dims = {
            'visual': config.get('visual_dim', 64),
            'auditory': config.get('auditory_dim', 64),
            'tactile': config.get('tactile_dim', 64),
            'olfactory': config.get('olfactory_dim', 64)
        }
        self.fusion_method = config.get('fusion_method', 'attention')
        self.output_dim = max(config.get('holistic_output_dim', 256), 1)  # Ensure it's at least 1
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    

        # Initialize fusion layers
        self.fusion_layers = {
            key: torch.nn.Linear(dim, self.output_dim).to(self.device)
            for key, dim in self.input_dims.items()
        }

        if self.fusion_method == 'attention':
            num_heads = min(config.get('num_heads', 4), self.output_dim)  # Ensure num_heads doesn't exceed output_dim
            # Ensure output_dim is divisible by num_heads
            self.output_dim = max((self.output_dim // num_heads) * num_heads, num_heads)
            self.attention = torch.nn.MultiheadAttention(self.output_dim, num_heads=num_heads).to(self.device)

        self.input_buffers = {key: deque(maxlen=config.get('buffer_size', 10)) for key in self.input_dims.keys()}

    def to(self, device):
        self.device = device
        for key in self.fusion_layers:
            self.fusion_layers[key] = self.fusion_layers[key].to(device)
        if hasattr(self, 'attention'):
            self.attention = self.attention.to(device)
        if hasattr(self, 'neural_fusion_model'):
            self.neural_fusion_model = self.neural_fusion_model.to(device)
        return self

    async def fuse(self, input_data):
        if not isinstance(input_data, dict) or set(input_data.keys()) != set(self.input_dims.keys()):
            raise ValueError("input_data must be a dictionary with keys: visual, auditory, tactile, olfactory")

        processed_inputs = {}
        for key, tensor in input_data.items():
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, dtype=torch.float32)
            tensor = tensor.to(self.device)
            processed_inputs[key] = self.fusion_layers[key](tensor)

        if self.fusion_method == 'concatenate':
            fused = torch.cat(list(processed_inputs.values()), dim=-1)
        elif self.fusion_method == 'sum':
            fused = sum(processed_inputs.values())
        elif self.fusion_method == 'average':
            fused = torch.mean(torch.stack(list(processed_inputs.values())), dim=0)
        elif self.fusion_method == 'attention':
            inputs_stack = torch.stack(list(processed_inputs.values()))
            fused, _ = self.attention(inputs_stack, inputs_stack, inputs_stack)
            fused = fused.mean(dim=0)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        fused = F.relu(fused)

        return fused



    def set_fusion_method(self, method):
        if method not in ['concatenate', 'sum', 'average', 'attention']:
            raise ValueError(f"Unsupported fusion method: {method}")
        self.fusion_method = method

    async def perceive(self, input_data: Dict[str, np.ndarray]) -> np.ndarray:
        return await self.integrate_inputs(input_data)

    async def integrate_inputs(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        for key, value in inputs.items():
            if key in self.input_buffers:
                self.input_buffers[key].append(value)
            else:
                logger.warning(f"Unrecognized input type: {key}")
        return await self.fuse_inputs()

    async def fuse_inputs(self) -> Optional[np.ndarray]:
        inputs = []
        for key, buffer in self.input_buffers.items():
            if len(buffer) > 0:
                inputs.append(np.mean(buffer, axis=0))

        if not inputs:
            logger.warning("No inputs available for fusion")
            return None


        try:
            if self.fusion_method == 'mean':
                return np.mean(inputs, axis=0)
            elif self.fusion_method == 'neural':
                if self.neural_fusion_model is None:
                    logger.warning("Neural fusion method selected but model is not initialized. Using mean fusion.")
                    return np.mean(inputs, axis=0)
                inputs_tensor = torch.tensor(np.array(inputs)).float()
                return self.neural_fusion_model(inputs_tensor).detach().numpy()
                    

            else:
                logger.warning(f"Unrecognized fusion method: {self.fusion_method}. Using mean fusion.")
                return np.mean(inputs, axis=0)
        except Exception as e:
            logger.error(f"Error during input fusion: {e}")
            return None

    async def weighted_fusion(self, weights: Dict[str, float]) -> Optional[np.ndarray]:
        weighted_inputs = []
        for key, buffer in self.input_buffers.items():
            if len(buffer) > 0 and key in weights:
                weighted_inputs.append(weights[key] * np.mean(buffer, axis=0))

        if not weighted_inputs:
            logger.warning("No inputs available for weighted fusion")
            return None

        return np.sum(weighted_inputs, axis=0) / sum(weights.values())

    def set_fusion_method(self, method: str, model: Optional[nn.Module] = None) -> None:
        self.fusion_method = method
        if method == 'neural':
            if model is None:
                input_dim = sum(len(buffer[0]) if len(buffer) > 0 else 0 for buffer in self.input_buffers.values())
                output_dim = self.config.get('fusion_output_dim', input_dim)
                self.neural_fusion_model = NeuralFusionModel(input_dim, output_dim)
            else:
                self.neural_fusion_model = model
        logger.info(f"Fusion method set to: {method}")

    def save_neural_model(self, path: str) -> None:
        if self.neural_fusion_model is not None:
            torch.save(self.neural_fusion_model.state_dict(), path)
            logger.info(f"Neural fusion model saved to {path}")
        else:
            logger.warning("No neural fusion model to save")

    def load_neural_model(self, path: str) -> None:
        if self.neural_fusion_model is not None:
            self.neural_fusion_model.load_state_dict(torch.load(path))
            logger.info(f"Neural fusion model loaded from {path}")
        else:
            logger.warning("Cannot load model: No neural fusion model initialized")


# Example usage
if __name__ == "__main__":
    hpm = HolisticPerception()
    
    visual = np.random.rand(10)
    auditory = np.random.rand(10)
    tactile = np.random.rand(10)
    olfactory = np.random.rand(10)

    hpm.integrate_inputs({'visual': visual, 'auditory': auditory, 'tactile': tactile, 'olfactory': olfactory})
    fused_output = hpm.fuse_inputs()
    print("Fused output:", fused_output)

    hpm.log_inputs()
    hpm.visualize_fused_output(fused_output)

    # Weighted fusion example
    weights = {'visual': 1, 'auditory': 1.5, 'tactile': 1, 'olfactory': 0.5}
    weighted_fused_output = hpm.weighted_fusion(weights)
    print("Weighted fused output:", weighted_fused_output)
    hpm.visualize_fused_output(weighted_fused_output)

    # Neural network-based fusion
    neural_model = NeuralFusionModel(input_dim=4*10, output_dim=10)  # Adjust input_dim based on actual input sizes
    hpm.set_fusion_method('neural', neural_model)
    neural_fused_output = hpm.fuse_inputs()
    print("Neural network fused output:", neural_fused_output)
    hpm.visualize_fused_output(neural_fused_output)
