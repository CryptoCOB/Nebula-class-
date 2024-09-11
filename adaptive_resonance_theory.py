import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
from typing import List, Dict, Any, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveResonanceTheory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.vigilance = config.get('initial_vigilance', 0.5)
        self.variant = config.get('variant', 'ART1')
        self.weights = np.random.rand(self.input_dim, self.output_dim)
        self.activations = None
        self.device = config.get('device', 'cpu')

    def to(self, device: str) -> None:
        self.device = device
        if isinstance(self.weights, np.ndarray):
            self.weights = torch.tensor(self.weights, dtype=torch.float32)
        self.weights = self.weights.to(device)

    async def train(self, inputs: np.ndarray, epochs: int = 1) -> None:
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            for input_vector in inputs:
                activation = await self.compute_activation(input_vector)
                resonance = self.check_resonance(activation)
                if resonance:
                    await self.update_weights(input_vector, activation)
                else:
                    await self.reset_weights(input_vector)
            self.log_epoch_summary(epoch)

    async def process(self, input_data: np.ndarray) -> np.ndarray:
        # Train ART on input_data
        await self.train(input_data, epochs=13)
        self.logger.debug("ART training completed within process method")

        # Compute activation for visualization and checking resonance
        activation = await self.compute_activation(input_data)

        # Check resonance and log the result
        resonance = self.check_resonance(activation)
        self.logger.debug(f"Resonance check result: {resonance}")

        # Visualize weights and activations if needed
        self.visualize_weights()
        self.visualize_activations()

        # Then categorize or further process the data
        result = await self.categorize(input_data)
        self.logger.debug(f"Processed output: {result}")
        
        return result


    async def compute_activation(self, input_vector: np.ndarray) -> np.ndarray:
        try:
            if self.variant == 'ART1':
                activation = np.dot(input_vector, self.weights)
            elif self.variant == 'ART2':
                activation = np.tanh(np.dot(input_vector, self.weights))
            else:
                raise ValueError(f"Unknown ART variant: {self.variant}")
            self.activations = activation
            logger.debug(f"Activation: {activation}")
            return activation
        except Exception as e:
            logger.error(f"Error in compute_activation: {e}")
            raise

    def check_resonance(self, activation: np.ndarray) -> bool:
        norm = np.linalg.norm(activation)
        resonance = norm > self.vigilance
        logger.debug(f"Activation norm: {norm}, Resonance: {resonance}")
        return resonance

    async def update_weights(self, input_vector: np.ndarray, activation: np.ndarray) -> None:
        self.weights += np.outer(input_vector, activation)
        logger.info(f"Updated weights: {self.weights}")

    async def reset_weights(self, input_vector: np.ndarray) -> None:
        self.weights = np.random.rand(self.input_dim, self.output_dim)
        logger.warning("Weights reset due to lack of resonance.")

    def adjust_vigilance(self, new_vigilance: float) -> None:
        self.vigilance = new_vigilance
        logger.info(f"Vigilance parameter adjusted to {self.vigilance}")

    def visualize_weights(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.imshow(self.weights, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title("Weights Visualization")
        plt.xlabel("Output Neurons")
        plt.ylabel("Input Features")
        plt.show()

    def visualize_activations(self) -> None:
        if self.activations is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(self.activations)
            plt.title("Activations Visualization")
            plt.xlabel("Output Neurons")
            plt.ylabel("Activation Value")
            plt.grid(True)
            plt.show()
        else:
            logger.warning("No activations to visualize.")

    def log_epoch_summary(self, epoch: int) -> None:
        logger.info(f"End of epoch {epoch + 1}:")
        logger.info(f"Current weights: {self.weights}")

    async def batch_train(self, input_batches: List[np.ndarray], epochs: int = 1) -> None:
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs} - Batch Training")
            for batch in input_batches:
                await self.train(batch, epochs=1)

    def save_model(self, file_path: str) -> None:
        np.save(file_path, self.weights)
        logger.info(f"Model weights saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        self.weights = np.load(file_path)
        logger.info(f"Model weights loaded from {file_path}")

    async def predict(self, input_data: np.ndarray) -> np.ndarray:
        try:
            activations = await self.compute_activation(input_data)
            return np.argmax(activations, axis=1)
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            raise

    async def get_patterns(self) -> np.ndarray:
        return self.weights.T

    async def categorize(self, input_data: np.ndarray) -> np.ndarray:
        return await self.predict(input_data)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "weights_shape": self.weights.shape,
            "vigilance": self.vigilance,
            "variant": self.variant,
            "device": self.device
        }