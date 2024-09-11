import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import autocast
import numpy as np
from torch import amp
import math
from transformers import AutoTokenizer
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import QasmSimulator, Aer
from qiskit.circuit.library import CDKMRippleCarryAdder
import dimod
from dwave.system import LeapHybridSampler
import matplotlib.pyplot as plt
from torchviz import make_dot
from typing import Dict, Any, List, Union, Tuple

import wandb

# Importing your custom modules
from SharedUtil import Hidden_LSTM, EnhancedTreeOfThought
from meta_learner import (
    FeedForward, ImageInputModule, MultiHeadAttention, 
    LMStudioInterface, TabularInputModule, TextInputModule, 
    AdaptiveResonanceTheory
)

class VisualizationWrapper:
    @staticmethod
    def create_figure(figsize=(10, 5)):
        return plt.figure(figsize=figsize)

    @staticmethod
    def show():
        plt.show()

class LRTracker:
    def __init__(self, optimizer, initial_lr):
        self.optimizer = optimizer
        self.current_lr = initial_lr

    def step(self):
        self.optimizer.step()
        self.current_lr = self.optimizer.param_groups[0]['lr']

    def get_lr(self):
        return self.current_lr

class DynamicBatchNorm1d(nn.Module):
    def __init__(self):
        super(DynamicBatchNorm1d, self).__init__()
        self.bn = None

    def forward(self, x):
        if self.bn is None or self.bn.num_features != x.size(1):
            self.bn = nn.BatchNorm1d(x.size(1)).to(x.device)
        return self.bn(x)

class AdvancedMetaLearner(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(AdvancedMetaLearner, self).__init__()
        self.config = config

        if not torch.cuda.is_available():
            raise RuntimeError("This model requires CUDA, but it's not available.")

        self.device = torch.device("cuda")
        self.lmstudio = LMStudioInterface(config)
        self.text_module = nn.DataParallel(TextInputModule(config)).to(self.device)
        self.image_module = nn.DataParallel(ImageInputModule(config)).to(self.device)
        self.tabular_module = nn.DataParallel(TabularInputModule(config)).to(self.device)

        # Shared Transformer Encoder
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

        self.bn1 = DynamicBatchNorm1d()
        self.bn2 = DynamicBatchNorm1d()
        self.data_management_lstm = Hidden_LSTM(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        
        self.scaler = amp.GradScaler(device='cuda')
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.distillation_temp = config.get('distillation_temperature', 2.0)
        self.distillation_alpha = config.get('distillation_alpha', 0.5)
        self.distillation_criterion = nn.KLDivLoss(reduction="batchmean").to(self.device)

        self.art = AdaptiveResonanceTheory(config)
        self.epochs = config.get('epochs', 10)
        
      
        self.logger = self.setup_logging()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.train_loader = None
        self.lr_tracker = LRTracker(self.optimizer, self.config['learning_rate'])
        self.valid_data_types = ["text", "image", "tabular", "combined"]
        self.initialize_weights()

    def setup_scheduler(self, total_steps: int):
        """Set up the learning rate scheduler."""
        warmup_steps = 1000
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_warmup_cosine_decay(warmup_steps, total_steps))
        
    def create_train_loader(self, config):
        text_data = torch.randint(0, config['text_vocab_size'], (config['batch_size'] * 10, 128))
        text_labels = torch.randint(0, config['output_dim'], (config['batch_size'] * 10,))
        text_dataset = TensorDataset(text_data, text_labels)
        self.train_loader = DataLoader(text_dataset, batch_size=config['batch_size'], shuffle=True)
        return self.train_loader
    
    def ensure_same_device(self, *tensors):
        return [t.to(self.device) for t in tensors]
    
    def setup_logging(self):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(self.config['log_file'])
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def print_model_summary(self):
        summary = f"AdvancedMetaLearner Summary:\n"
        summary += f"Total parameters: {sum(p.numel() for p in self.parameters())}\n"
        summary += f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}\n"
        summary += f"Device: {self.device}\n"
        return summary
    
    def quantum_arithmetic_encode(self, tokens, freqs, shots=1024):
        qc = QuantumCircuit(len(tokens))
        for i, token in enumerate(tokens):
            qc.x(i)  # Apply X gate to represent the token
        
        backend = QasmSimulator()
        job = backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        
        most_likely_state = max(counts, key=counts.get)
        encoded_value = int(most_likely_state, 2) / (2 ** len(tokens))
        
        return encoded_value

    async def quantum_arithmetic_decode(self, encoded_value, freqs, shots=1024):
        self.logger.debug("Starting quantum_arithmetic_decode")
        try:
            qc = QuantumCircuit(len(freqs))
            qr = QuantumRegister(len(freqs))
            adder = CDKMRippleCarryAdder(len(freqs))

            adder = adder.compose(adder)

            backend = QasmSimulator()
            transpiled_qc = transpile(qc, backend)
            job = backend.run(transpiled_qc, shots=shots)
            result = job.result()
            counts = result.get_counts(qc)

            decoded_probs = []
            for state, count in counts.items():
                prob = count / shots
                decoded_probs.append((int(state, 2), prob))

            decoded_tokens = [token for state, prob in sorted(decoded_probs, reverse=True) for token, (low, high) in freqs.items() if low <= prob < high]

            self.logger.debug(f"Decoded tokens: {decoded_tokens}")
            return decoded_tokens
        except Exception as e:
            self.logger.error(f"Error in quantum_arithmetic_decode: {e}")
            raise

    async def quantum_sparse_encode(self, tokens, vocab_size):
        self.logger.debug("Starting quantum_sparse_encode")
        try:
            qc = QuantumCircuit(vocab_size)
            qr = QuantumRegister(vocab_size)

            for token in tokens:
                qc.x(qr[token])

            backend = QasmSimulator()
            transpiled_qc = transpile(qc, backend)
            job = backend.run(transpiled_qc)
            result = job.result()
            quantum_state = result.get_statevector(qc)

            self.logger.debug("Quantum sparse encoding completed")
            return quantum_state
        except Exception as e:
            self.logger.error(f"Error in quantum_sparse_encode: {e}")
            raise

    async def quantum_sparse_decode(self, quantum_state, vocab_size):
        self.logger.debug("Starting quantum_sparse_decode")
        try:
            backend = Aer.get_backend('statevector_simulator')
            transpiled_qc = transpile(QuantumCircuit(quantum_state), backend)
            job = backend.run(transpiled_qc)
            result = job.result()
            statevector = result.get_statevector(QuantumCircuit(quantum_state))

            decoded_tokens = [i for i, amplitude in enumerate(statevector) if abs(amplitude) > 1e-6]

            self.logger.debug(f"Decoded tokens: {decoded_tokens}")
            return decoded_tokens
        except Exception as e:
            self.logger.error(f"Error in quantum_sparse_decode: {e}")
            raise

    async def quantum_huffman_encode(self, frequencies):
        self.logger.debug("Starting quantum_huffman_encode")
        try:
            sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
            tokens = [token for token, freq in sorted_freq]
            freqs = [freq for token, freq in sorted_freq]

            num_tokens = len(tokens)
            Q = {}

            for i in range(num_tokens):
                for j in range(i + 1, num_tokens):
                    Q[(i, j)] = freqs[i] * freqs[j]

            bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
            sampler = LeapHybridSampler()
            sampleset = sampler.sample(bqm, label='Example - Quantum Huffman Encoding')
            sample = sampleset.first.sample

            tree = []
            for (i, j), value in sample.items():
                if value and i < num_tokens and j < num_tokens:
                    tree.append((tokens[i], tokens[j]))

            def build_codes(node, prefix="", code={}):
                if isinstance(node, str):
                    code[node] = prefix
                else:
                    left, right = node
                    build_codes(left, prefix + '0', code)
                    build_codes(right, prefix + '1', code)
                return code

            while len(tree) > 1:
                left, right = tree[0], tree[1]
                tree = tree[2:]
                tree.append((left, right))
                tree.sort(key=lambda x: sum(freqs[tokens.index(subnode)] for subnode in x if isinstance(subnode, str)))

            codes = build_codes(tree[0])
            self.logger.debug(f"Huffman codes: {codes}")
            return codes
        except Exception as e:
            self.logger.error(f"Error in quantum_huffman_encode: {e}")
            raise
    
    async def save_state(self, path):
        state = {
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'data_management_lstm_state': self.data_management_lstm.get_state(),
            'config': self.config
        }
        torch.save(state, path)
        self.logger.info(f"Model state saved to {path}")

    async def load_state(self, path):
        self.logger.debug(f"Loading model state from {path}")
        try:
            state = torch.load(path, map_location=self.device)
            
            # Load config first
            self.config = state['config']
            
            # Reinitialize the model with the loaded config
            self.__init__(self.config)
            
            # Now load the state dict
            self.load_state_dict(state['model_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])
            self.scheduler.load_state_dict(state['scheduler_state'])
            self.data_management_lstm.set_state(state['data_management_lstm_state'])
            
            self.to(self.device)
            self.logger.info(f"Model state loaded from {path}")
        except KeyError as e:
            self.logger.error(f"Missing key in saved state: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Error loading model state: {e}", exc_info=True)
            raise

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
        ce_loss = F.cross_entropy(outputs.to(labels.device), labels)
        return ce_loss
        
    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], data_type: str) -> torch.Tensor:
        if data_type == 'combined':
            text_input, image_input, tabular_input = x
            text_embed = self.text_module(text_input)
            image_embed = self.image_module(image_input)
            tabular_embed = self.tabular_module(tabular_input)

            max_seq_len = max(embed.size(1) for embed in [text_embed, image_embed, tabular_embed])
            embeddings = [F.pad(embed, (0, 0, 0, max_seq_len - embed.size(1))) for embed in [text_embed, image_embed, tabular_embed]]
            
            x = torch.cat(embeddings, dim=1)
            x = self.neural_fusion_model(x)  # Fuse the embeddings using the neural fusion model
        elif data_type in ['text', 'image', 'tabular']:
            module = getattr(self, f"{data_type}_module")
            x = module(x)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Process the data through shared modules
        x = self.shared_encoder(x)
        #x = self.quantum_perception(x)  # Apply quantum perception
        #x = self.tree_of_thought(x)     # Apply tree of thought logic
        
        # Classification and output
        x = self.classifier(x.mean(dim=1))
        return torch.sigmoid(x)
    
    async def train_model(self, data_loader, data_type, epochs, accumulation_steps=4):
        wandb.init(project="advanced-meta-learner")
        self.train()
        try:
            for epoch in range(epochs):
                self.logger.debug(f"Training epoch {epoch + 1}/{epochs}")
                total_loss = 0
                num_batches = 0
                for batch_idx, batch in enumerate(data_loader):
                    inputs, labels = self._process_batch(batch, data_type)
                    with autocast('cuda',dtype=torch.float16):
                        outputs = self.forward(inputs, data_type='combined')
                        loss = self.combined_loss(outputs, labels)
                        loss = loss / accumulation_steps

                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(data_loader):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        self._clear_cuda_cache()  # Add clear cache here

                    total_loss += loss.item() * accumulation_steps
                    num_batches += 1

                    if batch_idx % 100 == 0:
                        self.logger.debug(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

                avg_loss = total_loss / num_batches
                accuracy = self._get_accuracy(data_loader, data_type)

                self.logger.info(f"Epoch {epoch + 1} completed with average loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")
                wandb.log({"epoch": epoch, "loss": avg_loss, "accuracy": accuracy})

                self.scheduler.step()

        except Exception as e:
            self.logger.error(f"Error during training: {e}", exc_info=True)
            raise

        finally:
            wandb.finish()


    def adapt(self, new_data_loader: DataLoader, data_type: str, epochs: int = 3, min_lr: float = 1e-6, max_lr: float = 1e-4, 
            use_compound_learn: bool = False, use_distill: bool = False, teacher_model=None):
        self.train()
        total_steps = len(new_data_loader) * epochs
        self.optimizer = torch.optim.Adam(self.parameters(), lr=max_lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=min_lr)

        try:
            if use_compound_learn:
                train_loaders = [new_data_loader]
                data_types = [data_type]
                self.compound_learn(train_loaders, data_types, epochs)
            elif use_distill and teacher_model:
                self.distill(new_data_loader, data_type, teacher_model, epochs)
            else:
                for epoch in range(epochs):
                    self.logger.debug(f"Adapting epoch {epoch + 1}/{epochs}")
                    for batch in new_data_loader:
                        inputs, labels = self._process_batch(batch, data_type)
                        self.optimizer.zero_grad()
                        with autocast('cuda', dtype=torch.float16):
                            outputs = self.forward(inputs, data_type)
                            loss = self.criterion(outputs, labels)
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.logger.debug(f"Batch loss: {loss.item()}")
                        self._clear_cuda_cache()  # Add clear cache here
                    self.scheduler.step()

                    epoch_loss = self._get_loss(new_data_loader, data_type)
                    epoch_accuracy = self._get_accuracy(new_data_loader, data_type)
                    self.logger.info(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")
        except Exception as e:
            self.logger.error(f"Error during adaptation: {e}", exc_info=True)
            raise

        self.save_model(f'adapted_model_epoch{epochs}.pt')


    def compound_learn(self, train_loaders: List[DataLoader], data_types: List[str], epochs: int = 5, min_lr: float = 1e-6, max_lr: float = 1e-4):
        self.train()
        total_steps = sum(len(loader) for loader in train_loaders) * epochs
        self.optimizer = torch.optim.Adam(self.parameters(), lr=max_lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_warmup_cosine_decay(warmup_steps=1000, total_steps=total_steps))

        def evaluate_batch_loss(data_loader, data_type):
            batch_losses = []
            self.eval()
            with torch.no_grad():
                for batch in data_loader:
                    inputs, labels = self._process_batch(batch, data_type)
                    outputs = self.forward(inputs, data_type)
                    loss = F.cross_entropy(outputs, labels)
                    batch_losses.append(loss.item())
            return batch_losses

        def organize_batches(train_loaders, data_types):
            all_batches = []
            for data_loader, data_type in zip(train_loaders, data_types):
                batch_losses = evaluate_batch_loss(data_loader, data_type)
                batches = list(data_loader)
                sorted_batches = [batch for _, batch in sorted(zip(batch_losses, batches), key=lambda x: x[0], reverse=True)]
                all_batches.extend([(batch, data_type) for batch in sorted_batches])
            return all_batches

        all_batches = organize_batches(train_loaders, data_types)

        def dynamic_training(batches, epoch):
            num_batches = len(batches)
            for i in range(1, num_batches + 1):
                indices = list(range(i, 0, -1)) + [0]
                for idx in indices:
                    batch, data_type = batches[idx]
                    inputs, labels = self._process_batch(batch, data_type)
                    self.optimizer.zero_grad()
                    with autocast(enabled=True, dtype=torch.float16):
                        outputs = self.forward(inputs, data_type)
                        loss = F.cross_entropy(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.logger.debug(f"Epoch {epoch + 1}, Batch {idx + 1} loss: {loss.item()}")
                    self._clear_cuda_cache()  # Add clear cache here

        try:
            for epoch in range(epochs):
                self.logger.debug(f"Compound learning epoch {epoch + 1}/{epochs}")
                dynamic_training(all_batches, epoch)
                self.scheduler.step()
                epoch_loss = self._get_loss(train_loaders[0], data_types[0])
                epoch_accuracy = self._get_accuracy(train_loaders[0], data_types[0])
                self.logger.info(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")

        except Exception as e:
            self.logger.error(f"Error during compound learning: {e}", exc_info=True)
            raise

        self.save_model(f'compound_learn_model_epoch{epochs}.pt')

    def distill(self, data_loader: DataLoader, data_type: str, epochs: int = 10):
        self.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            batch_count = 0
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    teacher_outputs = self.lmstudio.generate_outputs(inputs.cpu().tolist(), data_type)
                    teacher_outputs = torch.tensor(teacher_outputs, device=self.device)
                
                student_outputs = self(inputs, data_type)
                
                loss = self._calculate_distillation_loss(student_outputs, teacher_outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count
            self.logger.debug(f"Distillation Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            self.scheduler.step()

    def _calculate_distillation_loss(self, student_outputs, teacher_outputs, labels):
        distillation_loss = F.kl_div(
            F.log_softmax(student_outputs / self.config['distillation_temperature'], dim=1),
            F.softmax(teacher_outputs / self.config['distillation_temperature'], dim=1),
            reduction='batchmean'
        ) * (self.config['distillation_temperature'] ** 2)

        student_loss = F.cross_entropy(student_outputs, labels)

        total_loss = self.config['distillation_alpha'] * distillation_loss + (1 - self.config['distillation_alpha']) * student_loss

        return total_loss

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

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

    def _process_batch(self, batch, data_type):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, labels = batch
        else:
            raise ValueError(f"Expected batch to be a tuple or list of (inputs, labels), got {type(batch)}")
        
        if data_type == 'combined':
            if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
                text_inputs, image_inputs, tabular_inputs = inputs
                return (
                    text_inputs.to(self.device),
                    image_inputs.to(self.device),
                    tabular_inputs.to(self.device)
                ), labels.to(self.device)
            else:
                raise ValueError(f"Expected 3 input tensors for combined data, got {len(inputs)}")
        else:
            return inputs.to(self.device), labels.to(self.device)

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

    def load_model(self, path):
        self.logger.debug(f"Loading model from {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            model_state_dict = checkpoint['model_state_dict']
            
            # Remove unexpected keys
            unexpected_keys = ["bn1.bn.weight", "bn1.bn.bias", "bn1.bn.running_mean", "bn1.bn.running_var", "bn1.bn.num_batches_tracked",
                            "bn2.bn.weight", "bn2.bn.bias", "bn2.bn.running_mean", "bn2.bn.running_var", "bn2.bn.num_batches_tracked"]
            for key in unexpected_keys:
                model_state_dict.pop(key, None)
            
            self.load_state_dict(model_state_dict, strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config = checkpoint['config']
            self.to(self.device)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            
            raise


    async def receive_proposal(self, proposed_parameters):
        self.logger.info("Received proposal from EvolutionaryOptimizer")
        try:
            # Apply the proposed parameters to the model
            for name, param in self.named_parameters():
                if name in proposed_parameters:
                    param.data = torch.tensor(proposed_parameters[name], device=self.device)
            
            self.logger.info("Applied proposed parameters to the model")
            
            # Optionally, you can evaluate the model with these new parameters
            # and decide whether to keep them or revert to the previous state
            
            return True  # Indicating successful application of the proposal
        except Exception as e:
            self.logger.error(f"Error applying proposal: {e}")
            return False


    def refine(self, new_data_loader: DataLoader, data_type: str, min_lr: float = 1e-6, max_lr: float = 1e-4, epochs: int = 5):
        self.train()
        total_steps = len(new_data_loader) * epochs
        self.optimizer = torch.optim.Adam(self.parameters(), lr=max_lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=min_lr)

        try:
            for epoch in range(epochs):
                self.logger.debug(f"Refinement epoch {epoch + 1}/{epochs}")
                for batch in new_data_loader:
                    inputs, labels = self._process_batch(batch, data_type)
                    self.optimizer.zero_grad()
                    with autocast('cuda', dtype=torch.float16):
                        outputs = self.forward(inputs, data_type)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.logger.debug(f"Batch loss: {loss.item()}")
                    self._clear_cuda_cache()  # Add clear cache here
                self.logger.info(f"Refinement epoch {epoch + 1} completed")
        except Exception as e:
            self.logger.error(f"Error during refinement: {e}", exc_info=True)
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

    def visualize_weights(self):
        fig = VisualizationWrapper.create_figure()
        plt.imshow(self.weights, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title("Weights Visualization")
        plt.xlabel("Output Neurons")
        plt.ylabel("Input Features")
        VisualizationWrapper.show()

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
        'lmstudio_server_url': os.getenv('LM_STUDIO_API', 'http://127.0.0.1:1234'),
        'visual_dim': 768,
        'mutation_rate_start': 0.5,
        'mutation_rate_decay': 0.95,
    }

    text_data = torch.randint(0, config['text_vocab_size'], (8, 128))
    text_labels = torch.randint(0, config['output_dim'], (8,))
    text_dataset = TensorDataset(text_data, text_labels)
    text_loader = DataLoader(text_dataset, batch_size=2, shuffle=True)

    model = AdvancedMetaLearner(config)
    logger.info("AdvancedMetaLearner initialized")

    logger.info("Starting model training")
    model.train_model(text_loader, data_type='text', epochs=5)  # or whatever number of epochs you want
    
    logger.info("Starting model adaptation")
    model.adapt(text_loader, data_type='text', epochs=3)

    logger.info("Starting compound learning")
    model.compound_learn([text_loader], ['text'], epochs=3)

    teacher_model = AdvancedMetaLearner(config)  # This would be your actual teacher model
    logger.info("Starting distillation")
    model.distill(text_loader, data_type='text', epochs=3)

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
