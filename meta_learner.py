import json
import os
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchviz import make_dot
from torch.optim.lr_scheduler import LambdaLR
import math

import requests
import logging
import subprocess
import torch.multiprocessing as mp
from typing import Dict, Any, List, Tuple, Callable, Union
import numpy as np
import matplotlib.pyplot as plt

# Set the environment variable for CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Specify GPUs in order
auth_token = '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'

# Set the multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TextInputModule(nn.Module):
    def __init__(self, config):
        super(TextInputModule, self).__init__()
        self.embedding = nn.Embedding(config['text_vocab_size'], config['embed_dim'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.long())

class ImageInputModule(nn.Module):
    def __init__(self, config):
        super(ImageInputModule, self).__init__()
        self.config = config
        self.conv = nn.Conv2d(config['image_input_channels'], config['embed_dim'], kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((32, 32))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(config['embed_dim'] * 32 * 32, config['embed_dim'])
        self.relu = nn.ReLU()  # Add activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)  # Apply ReLU after convolution
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)  # Apply ReLU after the fully connected layer
        return x.unsqueeze(1)

    
class TabularInputModule(nn.Module):
    def __init__(self, config):
        super(TabularInputModule, self).__init__()
        self.config = config
        self.fc = nn.Linear(config['tabular_input_dim'], config['embed_dim'])
        self.relu = nn.ReLU()  # Add activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.relu(x)  # Apply ReLU after fully connected layer
        return x.unsqueeze(1)  # Add sequence dimension
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        input_dim = config['embed_dim']
        num_heads = config['num_heads']
        assert input_dim % num_heads == 0, "Input dimensions must be divisible by the number of heads."
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device  # Ensure the input tensor is on the right device
        N, seq_length, input_dim = x.shape
        
        # Move the model parameters to the same device as the input tensor
        queries = self.query(x).view(N, seq_length, self.num_heads, self.head_dim).to(device)
        keys = self.key(x).view(N, seq_length, self.num_heads, self.head_dim).to(device)
        values = self.value(x).view(N, seq_length, self.num_heads, self.head_dim).to(device)

        # Debugging shape info if needed
        assert queries.shape == keys.shape == values.shape, f"Shape mismatch: queries {queries.shape}, keys {keys.shape}, values {values.shape}"

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_length, input_dim)
        out = self.fc_out(out).to(device)

        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(config['embed_dim'], config['hidden_dim'])
        self.fc2 = nn.Linear(config['hidden_dim'], config['embed_dim'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device  # Ensure that the input tensor is on the right device
        x = self.fc1(x).to(device)  # Move intermediate result to the correct device
        x = F.relu(x)  # Apply activation
        x = self.fc2(x).to(device)  # Move final result to the correct device
        return x
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import requests
import torch
import base64
from openai import OpenAI

class LMStudioInterface:
    def __init__(self, server_url: str):
        server_url = 'http://127.0.0.1:1234'
        self.client = OpenAI(base_url=server_url, api_key="lm-studio")

    def generate_outputs(self, inputs, data_type):
        if data_type == 'text':
            return self._generate_text_outputs(inputs)
        elif data_type == 'image':
            return self._generate_image_outputs(inputs)
        else:
            raise ValueError("Unsupported data type. This interface supports 'text' and 'image' data types.")

    def _generate_text_outputs(self, input_texts):
        payload = {"texts": input_texts}
        response = requests.post(f"{self.server_url}/v1/embeddings", json=payload)

        if response.status_code != 200:
            raise Exception(f"Error from LM Studio server: {response.status_code}, {response.text}")

        logits = response.json()["logits"]
        logits_tensor = torch.tensor(logits)
        
        return logits_tensor

    def _generate_image_outputs(self, image_path):
        base64_image = ""
        try:
            image = open(image_path.replace("'", ""), "rb").read()
            base64_image = base64.b64encode(image).decode("utf-8")
        except:
            raise ValueError("Couldn't read the image. Make sure the path is correct and the file exists.")

        completion = self.client.chat.completions.create(
            model="lmstudio-community/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
            messages=[
                {
                    "role": "system",
                    "content": "This is a chat between a user and an assistant. The assistant is helping the user to describe an image.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
            stream=True
        )

        description = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                description += chunk.choices[0].delta.content
        
        return description

    
    
class AdaptiveResonanceTheory:
    def __init__(self, config):
        self.config = config
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.vigilance = config.get('initial_vigilance', 0.5)
        self.variant = config.get('variant', 'ART1')

        # Move weights to torch and initialize on the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = torch.rand(self.input_dim, self.output_dim, device=self.device)
        self.activations = None

    async def train(self, inputs: torch.Tensor, epochs: int = 1):
        inputs = inputs.to(self.device)  # Ensure inputs are on the same device
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            for input_vector in inputs:
                activation = self.compute_activation(input_vector)
                resonance = self.check_resonance(activation)
                if resonance:
                    self.update_weights(input_vector, activation)
                else:
                    self.reset_weights(input_vector)
            self.log_epoch_summary(epoch)

    def compute_activation(self, input_vector: torch.Tensor) -> torch.Tensor:
        input_vector = input_vector.to(self.device)  # Ensure input is on the correct device
        if self.variant == 'ART1':
            activation = torch.matmul(input_vector, self.weights)
        elif self.variant == 'ART2':
            activation = torch.tanh(torch.matmul(input_vector, self.weights))
        else:
            raise ValueError(f"Unknown ART variant: {self.variant}")
        self.activations = activation
        logging.debug(f"Activation: {activation}")
        return activation

    def check_resonance(self, activation: torch.Tensor) -> bool:
        norm = torch.norm(activation)
        resonance = norm > self.vigilance
        logging.debug(f"Activation norm: {norm}, Resonance: {resonance}")
        return resonance

    def update_weights(self, input_vector: torch.Tensor, activation: torch.Tensor):
        input_vector = input_vector.to(self.device)  # Ensure input is on the correct device
        self.weights += torch.outer(input_vector, activation)
        logging.info(f"Updated weights: {self.weights}")

    def reset_weights(self, input_vector: torch.Tensor):
        self.weights = torch.rand(self.input_dim, self.output_dim, device=self.device)
        logging.warning("Weights reset due to lack of resonance.")

    def adjust_vigilance(self, new_vigilance: float):
        self.vigilance = new_vigilance
        logging.info(f"Vigilance parameter adjusted to {self.vigilance}")

    def visualize_weights(self):
        weights_np = self.weights.cpu().numpy()  # Move weights to CPU for visualization
        plt.figure(figsize=(10, 5))
        plt.imshow(weights_np, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title("Weights Visualization")
        plt.xlabel("Output Neurons")
        plt.ylabel("Input Features")
        plt.show()

    def visualize_activations(self):
        if self.activations is not None:
            activations_np = self.activations.cpu().numpy()  # Move activations to CPU for visualization
            plt.figure(figsize=(10, 5))
            plt.plot(activations_np)
            plt.title("Activations Visualization")
            plt.xlabel("Output Neurons")
            plt.ylabel("Activation Value")
            plt.grid(True)
            plt.show()
        else:
            logging.warning("No activations to visualize.")

    def log_epoch_summary(self, epoch: int):
        logging.info(f"End of epoch {epoch + 1}:")
        logging.info(f"Current weights: {self.weights}")

    def batch_train(self, input_batches: list, epochs: int = 1):
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs} - Batch Training")
            for batch in input_batches:
                self.train(batch, epochs=1)

    def save_model(self, file_path: str):
        torch.save(self.weights.cpu(), file_path)  # Save weights as torch tensor
        logging.info(f"Model weights saved to {file_path}")

    def load_model(self, file_path: str):
        self.weights = torch.load(file_path, map_location=self.device)  # Load weights on the correct device
        logging.info(f"Model weights loaded from {file_path}")

def linear_warmup_cosine_decay(warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 0.5 * (1 + math.cos(math.pi * float(step - warmup_steps) / float(max(1e-6, total_steps - warmup_steps))))
        return lr_lambda

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import math
from typing import Dict, Any, List, Union, Tuple
from graphviz import Digraph

class AdvancedMetaLearner(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        if not torch.cuda.is_available():
            raise RuntimeError("This model requires CUDA, but it's not available.")

        self.device = torch.device("cuda")

        self.text_module = nn.DataParallel(TextInputModule(config)).to(self.device)
        self.image_module = nn.DataParallel(ImageInputModule(config)).to(self.device)
        self.tabular_module = nn.DataParallel(TabularInputModule(config)).to(self.device)
        self.art = AdaptiveResonanceTheory(config)
        
        # Increase the number of layers in the shared encoder
        num_layers = config.get('num_transformer_layers', 12)
        self.shared_encoder = nn.DataParallel(nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config['embed_dim'], nhead=config['num_heads'], batch_first=True),
            num_layers=num_layers
        )).to(self.device)

        self.attention = MultiHeadAttention(config).to(self.device)
        self.norm1 = nn.LayerNorm(config['embed_dim']).to(self.device)
        self.ffn = FeedForward(config).to(self.device)
        self.norm2 = nn.LayerNorm(config['embed_dim']).to(self.device)
        self.classifier = nn.Linear(config['embed_dim'], config['output_dim']).to(self.device)

        self.bn1 = nn.BatchNorm1d(config['embed_dim'])
        self.bn2 = nn.BatchNorm1d(config['hidden_dim'])

        self.optimizer = optim.Adam(self.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
        self.scaler = GradScaler()
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.distillation_temp = config.get('distillation_temperature', 2.0)
        self.distillation_alpha = config.get('distillation_alpha', 0.5)
        self.distillation_criterion = nn.KLDivLoss(reduction="batchmean").to(self.device)
        
        self.initialize_weights()
        self.logger = self.setup_logging()
        self.total_steps = len(self.train_loader) * self.epochs
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_warmup_cosine_decay(1000, self.total_steps))

    def setup_logging(self):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(f'{self.__class__.__name__}.log')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], data_type: str) -> torch.Tensor:
        if data_type == 'combined':
            text_input, image_input, tabular_input = x
            text_embed = self.text_module(text_input.to(self.device))
            image_embed = self.image_module(image_input.to(self.device))
            tabular_embed = self.tabular_module(tabular_input.to(self.device))
            
            max_seq_len = max(embed.size(1) for embed in [text_embed, image_embed, tabular_embed])
            embeddings = [F.pad(embed, (0, 0, 0, max_seq_len - embed.size(1))) for embed in [text_embed, image_embed, tabular_embed]]
            
            x = torch.cat(embeddings, dim=1)
        elif data_type in ['text', 'image', 'tabular']:
            module = getattr(self, f"{data_type}_module")
            x = module(x.to(self.device))
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        x = self.bn1(x)
        x = self.shared_encoder(x)
        x = self.attention(x)
        x = self.norm1(x)
        x = self.ffn(x)
        x = self.norm2(x)
        x = self.bn2(x)
        x = self.classifier(x.mean(dim=1))
        return torch.sigmoid(x)

    def linear_warmup_cosine_decay(self, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 0.5 * (1 + math.cos(math.pi * float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))))
        return lr_lambda

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def combined_loss(self, outputs, labels):
        ce_loss = F.cross_entropy(outputs, labels)
        smooth_l1_loss = F.smooth_l1_loss(outputs, F.one_hot(labels, num_classes=self.config['output_dim']).float())
        return ce_loss + 0.1 * smooth_l1_loss

    async def train_model(self, data_loader, data_type, epochs):
        self.train()
        try:
            for epoch in range(self.epochs):
                self.logger.debug(f"Training epoch {epoch + 1}/{self.epochs}")
                for batch in self.data_loader:
                    inputs, labels = self._process_batch(batch)
                    self.optimizer.zero_grad()
                    with autocast():
                        outputs = self.forward(inputs, data_type='combined')
                        loss = self.combined_loss(outputs, labels)
                    self.scaler.scale(loss).backward()
                    
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.logger.debug(f"Epoch {epoch + 1} completed with loss: {loss.item()}")
                self.scheduler.step()
        except Exception as e:
            self.logger.error(f"Error during training: {e}", exc_info=True)
            raise

    def adapt(self, new_data: DataLoader, data_type: str, epochs: int = 3):
        self.train()
        total_steps = len(new_data) * epochs
        warmup_steps = 1000
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_warmup_cosine_decay(warmup_steps, total_steps))
        try:
            for epoch in range(epochs):
                self.logger.debug(f"Adapting epoch {epoch + 1}/{epochs}")
                for batch in new_data:
                    inputs, labels = self._process_batch(batch, data_type)
                    self.optimizer.zero_grad()
                    with autocast():
                        outputs = self.forward(inputs, data_type)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.logger.debug(f"Batch loss: {loss.item()}")
                    self._clear_cuda_cache()
                self.scheduler.step()
        except Exception as e:
            self.logger.error(f"Error during adaptation: {e}", exc_info=True)
            raise

    def compound_learn(self, train_loaders: List[DataLoader], data_types: List[str], epochs: int = 5):
        self.train()
        total_steps = sum(len(loader) for loader in train_loaders) * epochs
        warmup_steps = 1000
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_warmup_cosine_decay(warmup_steps, total_steps))
        try:
            for epoch in range(epochs):
                self.logger.debug(f"Compound learning epoch {epoch + 1}/{epochs}")
                for data_loader, data_type in zip(train_loaders, data_types):
                    for batch in data_loader:
                        inputs, labels = self._process_batch(batch, data_type)
                        self.optimizer.zero_grad()
                        outputs = self.forward(inputs, data_type)
                        loss = F.cross_entropy(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                
                self.scheduler.step()
        except Exception as e:
            self.logger.error(f"Error during compound learning: {e}", exc_info=True)
            raise

    def distill(self, teacher_model: nn.Module, data_loader: DataLoader, data_type: str, epochs: int = 10):
        self.train()
        teacher_model.eval()
        teacher_model.to(self.device)
        total_steps = len(data_loader) * epochs
        warmup_steps = 1000
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_warmup_cosine_decay(warmup_steps, total_steps))
        try:
            for epoch in range(epochs):
                total_loss = 0.0
                batch_count = 0
                for batch in data_loader:
                    inputs, labels = self._process_batch(batch, data_type)
                    
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs, data_type)
                    
                    student_outputs = self.forward(inputs, data_type)
                    
                    if teacher_outputs.size(0) != student_outputs.size(0):
                        self.logger.warning(f"Mismatch in batch sizes. Teacher: {teacher_outputs.size(0)}, Student: {student_outputs.size(0)}")
                        min_batch_size = min(teacher_outputs.size(0), student_outputs.size(0))
                        teacher_outputs = teacher_outputs[:min_batch_size]
                        student_outputs = student_outputs[:min_batch_size]
                        labels = labels[:min_batch_size]
                    
                    loss = self._calculate_distillation_loss(student_outputs, teacher_outputs, labels)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                avg_loss = total_loss / batch_count
                self.logger.debug(f"Distillation Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
                self.scheduler.step()
        except Exception as e:
            self.logger.error(f"Error during distillation: {e}", exc_info=True)
            raise

    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, text_data, image_data, tabular_data, labels):
            self.text_data = text_data
            self.image_data = image_data
            self.tabular_data = tabular_data
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return (self.text_data[idx], self.image_data[idx], self.tabular_data[idx]), self.labels[idx]

    def _process_batch(self, batch, data_type='combined'):
        if data_type == 'combined':
            inputs, labels = batch
            text_inputs, image_inputs, tabular_inputs = inputs
            return (
                text_inputs.to(self.device),
                image_inputs.to(self.device),
                tabular_inputs.to(self.device)
            ), labels.to(self.device)
        else:
            inputs, labels = batch
            return inputs.to(self.device), labels.to(self.device)

    def _calculate_distillation_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.logger.debug("Calculating distillation loss")
        try:
            student_outputs = student_outputs.to(self.device)
            teacher_outputs = teacher_outputs.to(self.device)
            labels = labels.to(self.device)

            assert student_outputs.size() == teacher_outputs.size(), f"Mismatch in output sizes. Student: {student_outputs.size()}, Teacher: {teacher_outputs.size()}"

            soft_loss = self.distillation_criterion(
                F.log_softmax(student_outputs / self.distillation_temp, dim=1),
                F.softmax(teacher_outputs / self.distillation_temp, dim=1)
            ) * (self.distillation_temp ** 2)

            hard_loss = self.criterion(student_outputs, labels)

            return self.distillation_alpha * soft_loss + (1.0 - self.distillation_alpha) * hard_loss
        except Exception as e:
            self.logger.error(f"Error calculating distillation loss: {e}", exc_info=True)
            raise

    def _clear_cuda_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def predict(self, input_data: torch.Tensor, data_type: str) -> torch.Tensor:
        self.eval()
        try:
            with torch.no_grad():
                inputs = input_data.to(self.device)
                outputs = self.forward(inputs, data_type)
            return outputs
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}", exc_info=True)
            raise

    def save_model(self, path: str):
        self.logger.debug(f"Saving model to {path}")
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}", exc_info=True)
            raise

    def load_model(self, path: str):
        self.logger.debug(f"Loading model from {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config = checkpoint['config']
            self.to(self.device)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def visualize_model(self, input_tensor: torch.Tensor, filename: str):
        self.eval()
        try:
            with torch.no_grad():
                outputs = self.forward(input_tensor, data_type='text')
                dot = make_dot(outputs, params=dict(self.named_parameters()))
                dot.render(filename, format="png")
        except Exception as e:
            self.logger.error(f"Error during model visualization: {e}", exc_info=True)
            raise

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _log_gpu_usage(self, step_info: str):
        if self.device.type == 'cuda':
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                self.logger.info(f"Step: {step_info}")
                self.logger.info(f"GPU {i} memory allocated: {allocated} bytes")
                self.logger.info(f"GPU {i} memory reserved: {reserved} bytes")

    def _balanced_batch_loader(self, loader: DataLoader, device_ids: List[int]):
        device_count = len(device_ids)
        for i, (inputs, labels) in enumerate(loader):
            sub_batches = torch.chunk(inputs, device_count)
            sub_labels = torch.chunk(labels, device_count)
            for sub_input, sub_label, device_id in zip(sub_batches, sub_labels, device_ids):
                sub_input, sub_label = sub_input.to(f'cuda:{device_id}'), sub_label.to(f'cuda:{device_id}')
                yield sub_input, sub_label

    def _get_loss(self, data_loader, data_type):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = self._process_batch(batch, data_type)
                outputs = self.forward(inputs, data_type)
                loss = nn.functional.cross_entropy(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def _get_accuracy(self, data_loader, data_type):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = self._process_batch(batch, data_type)
                outputs = self.forward(inputs, data_type)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def to(self, device: torch.device):
        super().to(device)
        self.device = device
        return self

    async def train_art(self, inputs: np.ndarray, epochs: int = 1):
        self.art.train(inputs, epochs)
    
    def visualize_art_weights(self):
        self.art.visualize_weights()
    
    def visualize_art_activations(self):
        self.art.visualize_activations()

    def adjust_art_vigilance(self, new_vigilance: float):
        self.art.adjust_vigilance(new_vigilance)

    def save_art_model(self, file_path: str):
        self.art.save_model(file_path)

    def load_art_model(self, file_path: str):
        self.art.load_model(file_path)

    def diagnostics(self):
        return {
            'param_count': sum(p.numel() for p in self.parameters()),
            'layer_info': [layer for layer in self.modules()]
        }

# Example usage
def main():
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Starting AdvancedMetaLearner training and adaptation")

    config = {
        'log_file': os.getenv('LOG_FILE'),
        'auditory_dim': 768,
        'tactile_dim': 768,
        'olfactory_dim': 768,
        'fusion_dim': 768,
        'input_dim': 768,
        'output_dim': 768,
        'text_vocab_size': int(os.getenv('TEXT_VOCAB_SIZE')),
        'image_input_channels': 3,
        'tabular_input_dim': 768,
        'embed_dim': 768,
        'population_size': int(os.getenv('POPULATION_SIZE')),
        'learning_rate': float(os.getenv('LEARNING_RATE')),
        'batch_size': int(os.getenv('BATCH_SIZE')),
        'num_generations': int(os.getenv('NUM_GENERATIONS')),
        'reconstruction_weight': 1.0,
        'consistency_weight': 1.0,
        'info_preservation_weight': 1.0,
        'sparsity_weight': 1.0,
        'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
        'local_db_path': os.getenv('LOCAL_DB_PATH'),
        'root_state': 768,
        'lstm_input_dim': 768,
        'lstm_hidden_dim': 768,
        'lstm_output_dim': 768,
        'lstm_num_layers': int(os.getenv('LSTM_NUM_LAYERS')),
        'lstm_dropout': float(os.getenv('LSTM_DROPOUT')),
        'lstm_bidirectional': os.getenv('LSTM_BIDIRECTIONAL').lower() == 'true',
        'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME', 'default_index_name'),
        'pinecone_cloud': os.getenv('PINECONE_CLOUD'),
        'num_heads': int(os.getenv('NUM_HEADS')),
        'hidden_dim': 768,
        'ollama_server_url': os.getenv('OLLAMA_SERVER_URL', 'http://localhost:11434'),
        'visual_dim': 768,
        'mutation_rate_start': 0.5,
        'mutation_rate_decay': 0.95
    }

    text_data = torch.randint(0, config['text_vocab_size'], (8, 128))
    text_labels = torch.randint(0, config['output_dim'], (8,))
    text_dataset = TensorDataset(text_data, text_labels)
    text_loader = DataLoader(text_dataset, batch_size=2, shuffle=True)

    model = AdvancedMetaLearner(config, text_loader, epochs=5)
    logger.info("AdvancedMetaLearner initialized")

    logger.info("Starting model training")
    model.train_model()

    logger.info("Starting model adaptation")
    model.adapt(text_loader, data_type='text', epochs=3)

    logger.info("Starting compound learning")
    model.compound_learn([text_loader], ['text'], epochs=3)

    teacher_model = AdvancedMetaLearner(config, text_loader, epochs=5)  # This would be your actual teacher model
    logger.info("Starting distillation")
    model.distill(teacher_model, text_loader, data_type='text', epochs=3)

    logger.info("Visualizing AdvancedMetaLearner model")
    sample_input = text_data[0].unsqueeze(0)  # Adjust as needed for correct input shape
    model.visualize_model(sample_input, filename="advanced_meta_learner")

    logger.info("Starting another round of training after distillation")
    model.train_model()

    logger.info("Starting ART training")
    dummy_art_inputs = np.random.rand(10, config['input_dim'])
    model.train_art(dummy_art_inputs, epochs=5)

    logger.info("Visualizing ART weights")
    model.visualize_art_weights()

    logger.info("Visualizing ART activations")
    model.visualize_art_activations()

    logger.info("Adjusting ART vigilance")
    model.adjust_art_vigilance(0.7)

    logger.info("Saving ART model")
    model.save_art_model("art_model.npy")

    logger.info("Loading ART model")
    model.load_art_model("art_model.npy")

    logger.info("Running diagnostics")
    diagnostics = model.diagnostics()
    logger.info(f"Diagnostics: {diagnostics}")

    logger.info("AdvancedMetaLearner process completed")

if __name__ == "__main__":
    main()
