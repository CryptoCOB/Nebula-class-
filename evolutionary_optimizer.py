import os
import logging
import random
import traceback
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from multiprocessing import set_start_method

import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any
import datetime


# Set the start method to 'spawn' for multiprocessing
set_start_method('spawn', force=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the environment variable for CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"


from deap import creator, base, tools

# Define the creator for the DEAP evolutionary algorithm
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Create directories if they don't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

def evaluate_individual_wrapper(args: Tuple[Any, ...]) -> Tuple[float, float]:
    """Wrapper function for individual evaluation to be used with multiprocessing."""
    try:
        return evaluate_individual(*args)
    except Exception as e:
        logger.error(f"Failed to evaluate individual due to: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return a high fitness value to penalize this individual or handle as appropriate for your system
        return float('inf'), float('inf')  # Assuming lower fitness is better


def evaluate_individual(self, individual: List[torch.Tensor], 
                        input_dim: int, 
                        output_dim: int, 
                        training_data: Tuple[torch.Tensor, torch.Tensor], 
                        validation_data: Tuple[torch.Tensor, torch.Tensor], 
                        device: torch.device,
                        nas_architecture: Dict) -> Tuple[float, float]:
    """Evaluate an individual's fitness."""
    logger.debug(f"Evaluating individual with input_dim: {input_dim}, output_dim: {output_dim}")
    
    if nas_architecture:
        logger.debug(f"Using NAS architecture: {nas_architecture}")
        model = self.build_model_from_nas(nas_architecture).to(device)
    else:
        logger.debug("Using default model architecture")
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        ).to(device)
    
    logger.debug(f"Model architecture: {model}")

    def adjust_weights(layer: nn.Linear, individual_layer: torch.Tensor) -> torch.Tensor:
        layer_shape = layer.weight.data.shape
        ind_shape = individual_layer.shape
        logger.debug(f"Adjusting weights. Layer shape: {layer_shape}, Individual shape: {ind_shape}")
        if layer_shape != ind_shape:
            logger.warning(f"Shape mismatch. Layer: {layer_shape}, Individual: {ind_shape}")
            padded_layer = torch.zeros(layer_shape, device=device)
            padded_layer[:min(layer_shape[0], ind_shape[0]), :min(layer_shape[1], ind_shape[1])] = \
                individual_layer[:min(layer_shape[0], ind_shape[0]), :min(layer_shape[1], ind_shape[1])]
            logger.debug(f"Padded layer shape: {padded_layer.shape}")
            return padded_layer
        return individual_layer

    # Apply weights from individual to model
    linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    for i, (layer, ind_weight) in enumerate(zip(linear_layers, individual)):
        if layer.weight.data.shape == ind_weight.shape:
            layer.weight.data = ind_weight.clone().detach().to(device)
        else:
            logger.warning(f"Shape mismatch in layer {i}. Layer: {layer.weight.data.shape}, Individual: {ind_weight.shape}")
            layer.weight.data = adjust_weights(layer, ind_weight).to(device)

    # Training and evaluation logic
    train_texts, train_labels = training_data
    val_texts, val_labels = validation_data

    logger.debug(f"Training data shapes - Inputs: {train_texts.shape}, Labels: {train_labels.shape}")
    logger.debug(f"Validation data shapes - Inputs: {val_texts.shape}, Labels: {val_labels.shape}")

    train_data = TensorDataset(train_texts.clone().detach().float(), train_labels.clone().detach().float())
    val_data = TensorDataset(val_texts.clone().detach().float(), val_labels.clone().detach().float())

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Using Huggingface's Accelerator for distributed training across GPUs
    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    model.train()
    for epoch in range(5):  # Reduced number of epochs for faster evaluation
        train_loss = 0
        for inputs, labels in train_loader:
            torch.cuda.empty_cache()  # Clear cache to manage GPU memory
            optimizer.zero_grad()
            outputs = model(inputs)
            logger.debug(f"Batch - Input shape: {inputs.shape}, Output shape: {outputs.shape}, Label shape: {labels.shape}")
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item()
        logger.debug(f"Epoch {epoch+1}/5, Train Loss: {train_loss/len(train_loader)}")

    # Evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    fitness = val_loss / len(val_loader)
    complexity = torch.mean(torch.stack([torch.mean(layer ** 2) for layer in individual])).item()
    logger.debug(f"Individual evaluation complete. Fitness: {fitness}, Complexity: {complexity}")
    return fitness, complexity


def build_model_from_nas(self, nas_architecture: Optional[Dict]) -> nn.Module:
    """
    Build the model based on the NAS architecture.

    Args:
        nas_architecture: Dictionary defining the NAS architecture.

    Returns:
        An nn.Sequential model based on the provided NAS architecture.
    """
    if nas_architecture is None:
        # If no NAS architecture is provided, use a default architecture
        self.logger.info("No NAS architecture provided. Using default architecture.")
        return nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )
    
    layers = []
    
    # Iterate over the layers defined in the NAS architecture
    for layer_name, layer_info in nas_architecture.items():
        layer_type = layer_info.get('type', None)
        
        if layer_type == 'Linear':
            # Add Linear layer
            layers.append(nn.Linear(layer_info['in_features'], layer_info['out_features']))
            self.logger.debug(f"Added Linear layer: {layer_info['in_features']} -> {layer_info['out_features']}")
        
        elif layer_type == 'ReLU':
            # Add ReLU activation
            layers.append(nn.ReLU())
            self.logger.debug(f"Added ReLU activation.")
        
        elif layer_type == 'Dropout':
            # Add Dropout layer
            p = layer_info.get('p', 0.5)  # Default dropout probability is 0.5
            layers.append(nn.Dropout(p))
            self.logger.debug(f"Added Dropout with p={p}.")
        
        elif layer_type == 'BatchNorm':
            # Add Batch Normalization layer
            layers.append(nn.BatchNorm1d(layer_info['num_features']))
            self.logger.debug(f"Added BatchNorm1d with num_features={layer_info['num_features']}.")
        
        elif layer_type == 'Conv2d':
            # Add 2D Convolution layer if architecture specifies
            layers.append(nn.Conv2d(layer_info['in_channels'], layer_info['out_channels'], kernel_size=layer_info['kernel_size']))
            self.logger.debug(f"Added Conv2d layer: {layer_info['in_channels']} -> {layer_info['out_channels']} with kernel_size={layer_info['kernel_size']}")

        elif layer_type == 'MaxPool2d':
            # Add MaxPooling layer if required
            layers.append(nn.MaxPool2d(kernel_size=layer_info.get('kernel_size', 2)))
            self.logger.debug(f"Added MaxPool2d with kernel_size={layer_info.get('kernel_size', 2)}.")

        elif layer_type == 'Flatten':
            # Add a Flatten layer if necessary
            layers.append(nn.Flatten())
            self.logger.debug(f"Added Flatten layer.")

        else:
            # Log a warning if the layer type is not supported
            self.logger.warning(f"Unknown layer type in NAS architecture: {layer_type}")

    # Return the constructed model
    self.logger.info("Successfully built model from NAS architecture.")
    return nn.Sequential(*layers)


    
class EvolutionaryOptimizer:
    def __init__(self, input_dim: int, output_dim: int, training_data: Tuple[torch.Tensor, torch.Tensor], 
                 validation_data: Tuple[torch.Tensor, torch.Tensor], device: torch.device, 
                 population_size: int, mutation_rate_decay: float, mutation_rate_start: float, 
                 mutation_rate: float, nas_architecture: Optional[Dict] = None, 
                 use_multiprocessing: bool = True, **kwargs):
        
        # Use provided use_multiprocessing argument or default to True
        self.use_multiprocessing = use_multiprocessing
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing EvolutionaryOptimizer")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.training_data = training_data
        self.validation_data = validation_data
        self.device = device
        self.population_size = population_size
        self.mutate_rate = mutation_rate
        self.mutation_rate_start = mutation_rate_start
        self.mutation_rate_decay = mutation_rate_decay
        self.nas_architecture = nas_architecture

        self.logger.info(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
        self.logger.info(f"NAS architecture: {nas_architecture}")

        
        _, labels = training_data
        if len(labels.shape) == 1:
            # Handle 1D labels scenario, such as treating it as a single-class problem
            if output_dim != 1:
                self.logger.error(f"Expected output_dim of 1 for 1D labels, got {output_dim}")
                raise ValueError(f"Output dimension mismatch. Expected 1D labels for output_dim=1.")
        elif labels.shape[1] != output_dim:
            self.logger.warning(f"Output dimension mismatch detected. Expected: {output_dim}, Got: {labels.shape[1]}. Attempting to transpose labels.")
            # Attempt to transpose the labels
            labels = labels.T
            if labels.shape[0] == output_dim:
                self.logger.info("Successfully transposed labels to match the expected output dimension.")
            else:
                self.logger.error(f"Unable to resolve dimension mismatch. Expected: {output_dim}, Got after transpose: {labels.shape[1]}")
                raise ValueError(f"Output dimension mismatch. Check your data and configuration.")

        
        if self.use_multiprocessing:
            self.num_processes = torch.cuda.device_count()
        else:
            self.num_processes = os.cpu_count()

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.initialize_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.reproduce)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", evaluate_individual_wrapper)

        try:
            self.population = self.initialize_population()
            self.writer = SummaryWriter(log_dir='runs/evolutionary_optimizer')
            self.fitness_scores = self.evaluate_population(self.population)
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            raise
        self.elite_weights = []
        self.gen_handler = GenerationHandler(self)

    def setup_logging(self) -> logging.Logger:
        """Set up logging for the optimizer."""
        logger = logging.getLogger('EvolutionaryOptimizer')
        handler = logging.FileHandler('logs/evo_optimizer.log')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger

    async def optimize(self, parameters):
        """
        Optimize the parameters of the meta-learner using an evolutionary algorithm.

        Args:
            parameters (iterable): The parameters of the meta-learner.

        Returns:
            dict: The optimized proposal.
        """
        try:
            self.logger.info("Starting optimization with Evolutionary Optimizer.")

            # Evolve the population for a defined number of generations
            num_generations = 10  # You can adjust this value based on your needs
            for generation in range(num_generations):
                self.logger.info(f"Generation {generation + 1}/{num_generations}")
                
                # Evolve the population
                offspring = await self.evolve_population()

                # Clear cache after generating offspring
                torch.cuda.empty_cache()
                self.logger.debug("Cache cleared after generating offspring")

                # Combine current population and offspring, then select the next generation
                self.population = self.toolbox.select(self.population + offspring, self.population_size)

                # Clear cache after selection process
                torch.cuda.empty_cache()
                self.logger.debug(f"Cache cleared after population selection for generation {generation + 1}")

            # Extract the best individual
            best_individual = tools.selBest(self.population, 1)[0]
            self.logger.info("Optimization complete. Best individual selected.")

            # Log the best individual's details (optional)
            self.logger.debug(f"Best individual: {best_individual}")

            # Convert the best individual into a proposal format
            optimized_proposal = self._convert_to_proposal(best_individual)

            return optimized_proposal

        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            raise

        
    def to(self, device: torch.device):
        """Move the optimizer and its components to the specified device."""
        self.device = device
        if hasattr(self, 'model'):
            self.model.to(device)
        # Move other relevant components to the device
        self.logger.info(f"Moved EvolutionaryOptimizer to {device}")
        return self
    
    def initialize_individual_with_nas(self) -> Any:
        """Initialize a single individual based on the NAS architecture."""
        self.logger.debug("Initializing individual with NAS architecture")

        # Fall back to default initialization if no NAS architecture is provided
        if self.nas_architecture is None:
            return self.initialize_individual()

        layers = []
        try:
            for layer_name, layer_info in self.nas_architecture.items():
                # Ensure layer_info contains the required 'in_features' and 'out_features'
                if isinstance(layer_info, dict) and 'in_features' in layer_info and 'out_features' in layer_info:
                    # Initialize the layer with random weights on the specified device
                    layers.append(torch.randn(layer_info['out_features'], layer_info['in_features'], device=self.device))
                    self.logger.debug(f"Initialized layer {layer_name}: {layer_info}")
                else:
                    # Log a warning if the expected structure is not found
                    self.logger.warning(f"Unexpected layer info in NAS architecture: {layer_name}: {layer_info}")
            
            # Clear the cache after initializing all layers
            torch.cuda.empty_cache()
            self.logger.debug("Cache cleared after NAS-based individual initialization")
            
        except Exception as e:
            self.logger.error(f"Error initializing individual with NAS: {e}", exc_info=True)
            raise

        # Log the initialized layer shapes
        self.logger.debug(f"Individual initialized with NAS-based layer shapes: {[layer.shape for layer in layers]}")
        
        # Return the individual (ensure that creator.Individual is correctly defined elsewhere)
        return creator.Individual(layers)


        
    def evaluate_population(self, population: List[Any]) -> List[Tuple[float, float]]:
        """Evaluate the entire population using all available GPUs."""
        self.logger.debug("Evaluating population")

        fitnesses = []

        # Check if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for population evaluation")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Process each individual in the population
        for i, ind in enumerate(population):
            try:
                self.logger.debug(f"Evaluating individual {i+1}/{len(population)}")

                # Use the evaluate_individual_gpu function, which should handle multi-GPU execution
                fit = self.evaluate_individual_gpu(
                    ind, 
                    self.input_dim, 
                    self.output_dim, 
                    self.training_data, 
                    self.validation_data, 
                    device, 
                    self.nas_architecture
                )
                fitnesses.append(fit)

                # Clear the cache after evaluating each individual to free up GPU memory
                torch.cuda.empty_cache()
                self.logger.debug(f"Cache cleared after evaluating individual {i+1}")
                
            except Exception as e:
                self.logger.error(f"Error during GPU evaluation for individual {i+1}: {e}")
                raise

        # Assign fitness values back to the individuals
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        return fitnesses


    def build_model_from_nas(self, nas_architecture: Optional[Dict]) -> nn.Module:
        """
        Build the model based on the NAS architecture.

        Args:
            nas_architecture: Dictionary defining the NAS architecture.

        Returns:
            An nn.Sequential model based on the provided NAS architecture.
        """
        if nas_architecture is None:
            # If no NAS architecture is provided, use a default architecture
            self.logger.info("No NAS architecture provided. Using default architecture.")
            return nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_dim)
            )
        
        layers = []
        
        # Iterate over the layers defined in the NAS architecture
        for layer_name, layer_info in nas_architecture.items():
            layer_type = layer_info.get('type', None)
            
            if layer_type == 'Linear':
                # Add Linear layer
                layers.append(nn.Linear(layer_info['in_features'], layer_info['out_features']))
                self.logger.debug(f"Added Linear layer: {layer_info['in_features']} -> {layer_info['out_features']}")
            
            elif layer_type == 'ReLU':
                # Add ReLU activation
                layers.append(nn.ReLU())
                self.logger.debug(f"Added ReLU activation.")
            
            elif layer_type == 'Dropout':
                # Add Dropout layer
                p = layer_info.get('p', 0.5)  # Default dropout probability is 0.5
                layers.append(nn.Dropout(p))
                self.logger.debug(f"Added Dropout with p={p}.")
            
            elif layer_type == 'BatchNorm':
                # Add Batch Normalization layer
                layers.append(nn.BatchNorm1d(layer_info['num_features']))
                self.logger.debug(f"Added BatchNorm1d with num_features={layer_info['num_features']}.")
            
            elif layer_type == 'Conv2d':
                # Add 2D Convolution layer if architecture specifies
                layers.append(nn.Conv2d(layer_info['in_channels'], layer_info['out_channels'], kernel_size=layer_info['kernel_size']))
                self.logger.debug(f"Added Conv2d layer: {layer_info['in_channels']} -> {layer_info['out_channels']} with kernel_size={layer_info['kernel_size']}")

            elif layer_type == 'MaxPool2d':
                # Add MaxPooling layer if required
                layers.append(nn.MaxPool2d(kernel_size=layer_info.get('kernel_size', 2)))
                self.logger.debug(f"Added MaxPool2d with kernel_size={layer_info.get('kernel_size', 2)}.")

            elif layer_type == 'Flatten':
                # Add a Flatten layer if necessary
                layers.append(nn.Flatten())
                self.logger.debug(f"Added Flatten layer.")

            else:
                # Log a warning if the layer type is not supported
                self.logger.warning(f"Unknown layer type in NAS architecture: {layer_type}")

        # Return the constructed model
        self.logger.info("Successfully built model from NAS architecture.")
        return nn.Sequential(*layers)



    
    def initialize_population(self) -> List[Any]:
        """Initialize the population of individuals."""
        self.logger.debug("Initializing population")
        try:
            population = [self.initialize_individual() for _ in range(self.population_size)]
            self.logger.debug(f"Population initialized with {len(population)} individuals")
            return population
        except Exception as e:
            self.logger.error(f"Error during population initialization: {e}")
            raise
        
    def initialize_individual(self) -> Any:
        """Initialize a single individual."""
        self.logger.debug("Initializing individual")
        try:
            layers = []

            # Case 1: NAS architecture is provided
            if self.nas_architecture:
                for i in range(len(self.nas_architecture) - 1):
                    layer_info = self.nas_architecture.get(f'layer_{i+1}', {})
                    layer_type = layer_info.get('type')

                    # Handle layers that have in_features and out_features (e.g., Linear)
                    if layer_type == 'Linear':
                        in_features = layer_info['in_features']
                        out_features = layer_info['out_features']

                        # Initialize the layer with random weights
                        layers.append(torch.randn(out_features, in_features, device=self.device))
                        self.logger.debug(f"Initialized Linear layer {i}: {in_features} -> {out_features}")

                    # Handle non-linear layers like ReLU, Dropout, etc.
                    elif layer_type == 'ReLU':
                        layers.append(nn.ReLU())
                        self.logger.debug(f"Initialized ReLU layer {i}")

                    elif layer_type == 'Dropout':
                        p = layer_info.get('p', 0.5)  # Default dropout probability is 0.5
                        layers.append(nn.Dropout(p))
                        self.logger.debug(f"Initialized Dropout layer {i} with p={p}")

                    elif layer_type == 'BatchNorm':
                        num_features = layer_info.get('num_features', 256)  # Default number of features
                        layers.append(nn.BatchNorm1d(num_features))
                        self.logger.debug(f"Initialized BatchNorm layer {i} with {num_features} features")

                    else:
                        # Log a warning if the layer type is unknown
                        self.logger.warning(f"Unknown layer type at {i}: {layer_type}. Skipping layer.")

                torch.cuda.empty_cache()  # Clear the cache after the loop

            # Case 2: Default initialization if NAS architecture is not provided
            else:
                layers = [
                    torch.randn(256, self.input_dim, device=self.device),
                    torch.randn(128, 256, device=self.device),
                    torch.randn(64, 128, device=self.device),
                    torch.randn(self.output_dim, 64, device=self.device)
                ]
                torch.cuda.empty_cache()  # Clear cache after initialization

            # Log the initialized layers
            self.logger.debug(f"Individual initialized with layer shapes: {[layer.shape if isinstance(layer, torch.Tensor) else layer for layer in layers]}")

            # Return the individual (ensure creator.Individual is correctly defined elsewhere)
            return creator.Individual(layers)

        except Exception as e:
            # Log and raise any exceptions that occur
            self.logger.error(f"Error initializing individual: {e}")
            raise



    def evaluate_individual_gpu(self, individual: List[torch.Tensor], 
                                input_dim: int, 
                                output_dim: int, 
                                training_data: Tuple[torch.Tensor, torch.Tensor], 
                                validation_data: Tuple[torch.Tensor, torch.Tensor], 
                                device: torch.device,
                                nas_architecture: Any) -> Tuple[float, float]:
        """
        Evaluate an individual using a GPU-parallelized model.
        """
        self.logger.debug("Evaluating individual on GPU")
        
        try:
            # Step 1: Build the model from NAS architecture.
            model = self.build_model_from_nas(nas_architecture).to(device)
            self.logger.debug(f"Model structure: {model}")
            
            # If multiple GPUs are available, use DataParallel
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            
            # Step 2: Apply the individual's weights to the model.
            self.logger.debug("Applying weights to model")
            self._apply_weights_to_model(model, individual, nas_architecture)
            
            # Step 3: Prepare the data loaders.
            train_loader = self._prepare_data_loader(training_data, batch_size=32)
            val_loader = self._prepare_data_loader(validation_data, batch_size=32)
            
            # Step 4: Train the model.
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            model.train()
            for epoch in range(5):  # You can adjust the number of epochs
                self.logger.debug(f"Epoch {epoch + 1}/5")
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device).float(), targets.to(device).float()
                    optimizer.zero_grad()
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            # Step 5: Evaluate the model.
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    self.logger.debug(f"Validation batch - Inputs: {inputs.shape}, Targets: {targets.shape}")
                    inputs, targets = inputs.to(device).float(), targets.to(device).float()
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
            
            # Calculate fitness (lower is better).
            fitness = total_loss / len(val_loader)
            
            # Calculate complexity (e.g., L2 norm of weights).
            # Step 7: Calculate complexity, skipping non-parametric layers
            complexity = sum(
                torch.norm(weight).item() for weight in individual if isinstance(weight, torch.Tensor) and weight.dim() > 1
            )

            
            self.logger.debug(f"Individual evaluation complete. Fitness: {fitness}, Complexity: {complexity}")
            return fitness, complexity
        
        except Exception as e:
            self.logger.error(f"Error in GPU evaluation: {e}")
            self.logger.error(f"Error traceback: {traceback.format_exc()}")
            raise


    def _apply_weights_to_model(self, model: nn.Module, individual: List[torch.Tensor], nas_architecture: Dict):
        """Apply the weights from the individual to the model."""

        param_idx = 0
        self.logger.debug(f"Starting to apply weights. Individual length: {len(individual)}")
        
        device = next(model.parameters()).device

        for name, module in model.named_modules():
            self.logger.debug(f"Processing module: {name}, type: {type(module)}")
            
            if isinstance(module, (nn.Linear, nn.Conv2d)):  # Add other weight-bearing layers as needed
                self.logger.debug(f"Found parametric layer: {name}")
                
                try:
                    if name not in nas_architecture or nas_architecture[name]['type'] != type(module).__name__:
                        self.logger.warning(f"Layer {name} not in or mismatched with NAS architecture. Skipping.")
                        continue

                    expected_in, expected_out = nas_architecture[name]['in_features'], nas_architecture[name]['out_features']
                    if isinstance(module, nn.Linear):
                        assert module.in_features == expected_in and module.out_features == expected_out, f"Dimension mismatch in {name}"
                    
                    # Apply weights
                    if hasattr(module, 'weight'):
                        self._apply_parameter(module.weight, individual, param_idx, name, 'weight', device)
                        param_idx += 1
                    
                    # Apply bias if exists
                    if hasattr(module, 'bias') and module.bias is not None:
                        self._apply_parameter(module.bias, individual, param_idx, name, 'bias', device)
                        param_idx += 1
                except AssertionError as e:
                    self.logger.error(f"Assertion Error in {name}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error applying weights to {name}: {e}", exc_info=True)
            else:
                self.logger.debug(f"Skipping non-parametric layer or unsupported type: {name}")

        self.logger.debug(f"Finished applying weights. Used {param_idx} out of {len(individual)} weights")
        if param_idx < len(individual):
            self.logger.warning(f"Not all weights from the individual were applied. Applied {param_idx} out of {len(individual)}")

    def _apply_parameter(self, param, individual, idx, name, param_type, device):
        if idx < len(individual):
            ind_param = individual[idx].to(device)
            if param.data.shape == ind_param.shape:
                param.data.copy_(ind_param)
                self.logger.debug(f"Applied {param_type} to {name}")
                return True
            else:
                self.logger.warning(f"Shape mismatch for {name}.{param_type}: model {param.data.shape}, individual {ind_param.shape}")
                # Optionally, implement padding or cropping here if it makes sense for your application
                # For now, we'll just log and not apply
                return False
        return False  # Index out of range for individual



    def _prepare_data_loader(self, data: Tuple[torch.Tensor, torch.Tensor], batch_size: int) -> DataLoader:
        """Prepare a DataLoader from the given data."""
        inputs, targets = data
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def mutate(self, individual: Any, indpb: float = 0.1) -> Tuple[Any]:
        """Mutate an individual."""
        self.logger.debug("Mutating individual")
        for layer in individual:
            if torch.rand(1).item() < indpb:
                layer += torch.randn_like(layer) * 0.1
        self.logger.debug("Mutation complete")
        return individual,

    def _convert_to_proposal(self, best_individual):
        """
        Convert the best individual into a proposal dictionary.

        Args:
            best_individual (list): The best individual from the population.

        Returns:
            dict: The optimized proposal.
        """
        proposal = {"layer_{}".format(i + 1): layer.tolist() for i, layer in enumerate(best_individual)}
        return proposal
    

    def reproduce(self, ind1: Any, ind2: Any) -> Tuple[Any, Any]:
        """Perform mating between two individuals across multiple GPUs."""
        
        # Assign devices based on available GPUs
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        if len(devices) < 2:
            self.logger.error("Need at least 2 GPUs for multi-GPU reproduction")
            return ind1, ind2
        
        # Move individuals to different GPUs (one per individual)
        ind1.to(devices[0])
        ind2.to(devices[1])
        
        self.logger.debug(f"Models are Mating on {devices[0]} and {devices[1]}")
        
        # Perform layer-wise mating on each GPU
        for idx, (layer1, layer2) in enumerate(zip(ind1.layers, ind2.layers)):
            if torch.rand(1).item() < 0.5:
                ind1.layers[idx], ind2.layers[idx] = layer2, layer1
                self.logger.debug(f"Swapped layer {idx} between individuals")

        self.logger.debug("Mating complete")
        
        # Clear cache on both GPUs
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                self.logger.info(f"Cleared GPU cache on {device} after reproduction")
        
        return ind1, ind2


    def evolve(self, num_generations: int, device: torch.device):
        """Evolve the population for a specified number of generations across multiple GPUs."""
        
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        if len(devices) == 0:
            self.logger.error("No GPUs available for evolution.")
            return
        
        self.logger.info(f"Starting evolution for {num_generations} generations across GPUs {devices}")
        
        for gen in range(num_generations):
            # Assign specific GPUs for each generation to balance load
            active_device = devices[gen % len(devices)]
            self.logger.info(f"Running generation {gen + 1} on {active_device}")

            # Call evolution handler on specific device
            self.gen_handler.evolve(gen, num_generations, device=active_device)
            
            # Clear GPU cache for the active device
            with torch.cuda.device(active_device):
                torch.cuda.empty_cache()
                self.logger.info(f"Cleared GPU cache on {active_device} after generation {gen + 1}")
        
        self.logger.info("Evolution completed")


    async def evolve_population(self):
        """Evolve the population for one generation, respecting the NAS architecture."""
        try:
            offspring = self.toolbox.select(self.population, len(self.population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if torch.rand(1).item() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if torch.rand(1).item() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = await self.evaluate_population(invalid_ind)

            # Update mutation rate
            self.mutation_rate *= self.mutation_rate_decay

            self.logger.info(f"Evolved population. New offspring: {len(offspring)}, Invalid individuals: {len(invalid_ind)}")
            return offspring

        except Exception as e:
            self.logger.error(f"Error during population evolution: {e}")
            raise
        
    def load_state_dict(self) -> dict:
        """Get the current state of the optimizer."""
        self.logger.debug("Getting optimizer state")
        return {
            'population': self.population,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'generation': self.gen_handler.current_generation
        }

    def save_state_dict(self, state: dict):
        """Set the state of the optimizer."""
        self.logger.debug("Setting optimizer state")
        self.population = state['population']
        self.input_dim = state['input_dim']
        self.output_dim = state['output_dim']
        self.gen_handler.current_generation = state['generation']
        self.logger.debug("Optimizer state updated")

    async def propose_parameters(self, meta_learner):
        """Propose optimized parameters for the meta-learner by evolving the population."""
        try:
            self.logger.info("Proposing parameters using Evolutionary Optimizer with NAS integration")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            # Run the evolutionary algorithm
            self.evolve(num_generations=10)

            best_individual = tools.selBest(self.population, k=1)[0]

            # Extract and format the proposed parameters
            proposed_parameters = {}
            for i, layer in enumerate(best_individual):
                proposed_parameters[f"layer_{i+1}"] = layer.to(self.device).detach().numpy()

            self.logger.info(f"Proposed parameters based on NAS architecture: {proposed_parameters.keys()}")

            # Communicate the proposal to the meta-learner
            await meta_learner.receive_proposal(proposed_parameters)
            self.logger.info("Parameters communicated to meta-learner successfully.")

            return proposed_parameters

        except Exception as e:
            self.logger.error(f"Error in proposing parameters: {e}")
            raise
        
    def update_architecture(self, nas_architecture):
        self.logger.debug(f"Updating architecture: {nas_architecture}")
        self.nas_architecture = nas_architecture
        # Reinitialize the population based on the new architecture
        self.population = self.initialize_population()
        self.logger.info("Population reinitialized with new architecture")
        
class GenerationHandler:
    def __init__(self, evolutionary_optimizer: EvolutionaryOptimizer):
        self.evolutionary_optimizer = evolutionary_optimizer
        self.logger = evolutionary_optimizer.logger
        self.current_generation = 0
        self.mutation_rate = evolutionary_optimizer.mutation_rate_start
        self.mutation_rate_decay = evolutionary_optimizer.mutation_rate_decay
        self.elite_folder = "elites"
        self.log_folder = "logs"
        self.setup_folders()

    def setup_folders(self):
        if not os.path.exists(self.elite_folder):
            os.makedirs(self.elite_folder)
            self.logger.info(f"Created folder: {self.elite_folder}")
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
            self.logger.info(f"Created folder: {self.log_folder}")

    def save_elites(self, elites, generation):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, elite in enumerate(elites):
            elite_path = os.path.join(self.elite_folder, f"elite_gen{generation}_rank{i+1}_{timestamp}.pth")
            with open(elite_path, 'wb') as f:
                torch.save(elite, f)
            self.logger.info(f"Saved elite {i+1} of generation {generation} to {elite_path}")

    def evolve(self, generation: int, max_generations: int):
        """Evolve the population for one generation."""
        self.logger.debug(f"Evolving generation {generation}")
        
        # Selection
        offspring = self.evolutionary_optimizer.toolbox.select(
            self.evolutionary_optimizer.population, 
            len(self.evolutionary_optimizer.population)
        )
        offspring = list(map(self.evolutionary_optimizer.toolbox.clone, offspring))
        self.logger.debug(f"Selected {len(offspring)} offspring for next generation")

        # Elite selection
        elite_size = int(0.1 * len(self.evolutionary_optimizer.population))
        elites = tools.selBest(self.evolutionary_optimizer.population, elite_size)
        self.logger.debug(f"Selected {len(elites)} elite individuals")

        # Incentive for higher-scoring elites
        if elites:
            top_elite = elites[0]
            top_elite_count = 2  # Allow the top elite to reproduce more
            for _ in range(top_elite_count):
                offspring.append(self.evolutionary_optimizer.toolbox.clone(top_elite))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if torch.rand(1).item() < 0.5:
                self.evolutionary_optimizer.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        self.logger.debug("Crossover operations completed")

        # Random possibility of twins
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.1:  # 10% chance of twins
                offspring.append(self.evolutionary_optimizer.toolbox.clone(offspring[i]))
        self.logger.debug("Twins possibility applied")

        # Mutation
        mutation_rate = self.mutation_rate * (1 - generation / max_generations)
        for mutant in offspring:
            if random.random() < mutation_rate:
                self.evolutionary_optimizer.toolbox.mutate(mutant)
                del mutant.fitness.values
        self.logger.debug("Mutation operations completed")
        self.mutation_rate *= self.mutation_rate_decay

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.evolutionary_optimizer.evaluate_population(invalid_ind)
        self.logger.debug(f"Evaluated {len(invalid_ind)} invalid individuals")

        # Update population
        self.evolutionary_optimizer.population[:] = elites + offspring[:-elite_size]

        # Save elites
        self.save_elites(elites, generation)

        # Log progress
        best_fitness = min(ind.fitness.values[0] for ind in self.evolutionary_optimizer.population)
        self.evolutionary_optimizer.writer.add_scalar('Best Fitness', best_fitness, generation)
        self.logger.info(f"Generation {generation}: Best Fitness = {best_fitness}")

        self.current_generation += 1
        self.logger.debug(f"Generation {generation} evolution completed")
        
    def get_state(self):
        # Return the state dictionary for Evolutionary Optimizer
        return self.state_dict()

    def set_state(self, state_dict):
        # Set the state dictionary for Evolutionary Optimizer
        self.load_state_dict(state_dict)
        
import logging
import torch
import torch.nn as nn
from bayes_opt import BayesianOptimization
from torch import amp
from torch.utils.data import DataLoader, TensorDataset
import time


class ContextualLogger:
    def __init__(self, logger, context):
        self.logger = logger
        self.context = context

    def debug(self, message):
        self.logger.debug(f"[{self.context}] {message}")

    def info(self, message):
        self.logger.info(f"[{self.context}] {message}")

    def warning(self, message):
        self.logger.warning(f"[{self.context}] {message}")

    def error(self, message):
        self.logger.error(f"[{self.context}] {message}")

    def critical(self, message):
        self.logger.critical(f"[{self.context}] {message}")


class NeuralArchitectureSearch(nn.Module):
    def __init__(self, input_dim, output_dim, device=None, layer_units=None):
        super(NeuralArchitectureSearch, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = amp.GradScaler(device='cuda') 

        self.logger = ContextualLogger(self.setup_logging(), 'NeuralArchitectureSearch')
        self.log_health_check()
        layer_units = [64,128,256,512,1024]
        self.layers = self.initialize_layers(input_dim, output_dim, layer_units)

    def setup_logging(self):
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger('NeuralArchitectureSearch')
        handler = logging.FileHandler('nas.log')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def log_health_check(self):
        self.logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name(self.device)}")
            self.logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(self.device)} bytes")
            self.logger.info(f"Memory Cached: {torch.cuda.memory_reserved(self.device)} bytes")
        else:
            self.logger.info("CUDA is not available. Running on CPU.")

    def initialize_layers(self, input_dim, output_dim, layer_units: List[int]):
        """
        Initialize architecture with a dynamic number of layers based on `layer_units`.
        
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            layer_units (List[int]): A list containing the number of units in each hidden layer.
        
        Returns:
            nn.ModuleList: A list of initialized layers.
        """
        self.logger.debug(f"Initializing architecture with input_dim: {input_dim}, output_dim: {output_dim}, layer_units: {layer_units}")
        layers = []
        
        # Create input layer
        layers.append(nn.Linear(input_dim, layer_units[0]))
        layers.append(nn.ReLU())
        
        # Create hidden layers
        for i in range(1, len(layer_units)):
            layers.append(nn.Linear(layer_units[i-1], layer_units[i]))
            layers.append(nn.ReLU())
        
        # Create output layer
        layers.append(nn.Linear(layer_units[-1], output_dim))
        self.log_health_check()
        
        return nn.ModuleList(layers).to(self.device)
    


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def search(self, training_data, validation_data, num_layers=3):
        """
        Perform Neural Architecture Search with a flexible number of layers.

        Args:
            training_data: The training dataset.
            validation_data: The validation dataset.
            num_layers (int): Number of hidden layers to optimize (default is 3).

        Returns:
            None: Updates `self.layers` with the optimized architecture.
        """
        self.logger.debug("Starting Neural Architecture Search")

        # Dynamic bounds for each layer based on the number of layers
        pbounds = {f'layer{i}_units': (32, 512) for i in range(1, num_layers + 1)}

        def black_box_function(**layer_units):
            layer_units_values = [int(layer_units[f'layer{i}_units']) for i in range(1, num_layers + 1)]
            self.logger.debug(f"Evaluating architecture with layer units: {layer_units_values}")
            return -self.nas_algorithm(training_data, validation_data, layer_units_values)

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            random_state=1,
        )
        optimizer.maximize(init_points=10, n_iter=50)

        best_params = optimizer.max['params']
        self.logger.debug(f"Best parameters found: {best_params}")

        best_layer_units = [int(best_params[f'layer{i}_units']) for i in range(1, num_layers + 1)]
        self.layers = self.build_custom_model(self.input_dim, best_layer_units, self.output_dim)

    def build_custom_model(self, input_dim, layer_units: List[int], output_dim):
        """
        Build a custom model based on the number of layers and units.

        Args:
            input_dim (int): Number of input features.
            layer_units (List[int]): List of units in each hidden layer.
            output_dim (int): Number of output features.

        Returns:
            nn.ModuleList: A list of initialized layers.
        """
        self.logger.debug(f"Building custom model with input_dim: {input_dim}, layer_units: {layer_units}, output_dim: {output_dim}")
        model_layers = []

        # Create the input layer
        model_layers.append(nn.Linear(input_dim, layer_units[0]))
        model_layers.append(nn.ReLU())

        # Create hidden layers
        for i in range(1, len(layer_units)):
            model_layers.append(nn.Linear(layer_units[i-1], layer_units[i]))
            model_layers.append(nn.ReLU())

        # Create the output layer
        model_layers.append(nn.Linear(layer_units[-1], output_dim))

        self.logger.debug(f"Built custom model layers: {model_layers}")
        return nn.ModuleList(model_layers).to(self.device)


    def generate(self, input_data):
        self.logger.debug(f"Generating output with input data length: {len(input_data)}")
        try:
            start_time = time.time()
            input_tensor = input_data.clone().detach().to(self.device).float()
            self.logger.debug(f"Input tensor shape: {input_tensor.shape}")
            
            # Forward pass through the model
            output = self.forward(input_tensor)
            
            self.logger.debug(f"Output tensor shape: {output.shape}")
            end_time = time.time()
            self.logger.info(f"Generation completed in {end_time - start_time} seconds")
            return output.cpu().detach().numpy()
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise


    def nas_algorithm(self, train_loader, val_loader, layer_units: List[int]):
        """
        Runs the NAS algorithm for a model with flexible layer units.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            layer_units: List of integers representing the number of units in each hidden layer.
            
        Returns:
            float: Validation loss after training.
        """
        self.logger.debug(f"Running NAS algorithm with layer units: {layer_units}")
        
        try:
            # Build the model dynamically based on the layer_units
            layers = []
            layers.append(nn.Linear(self.input_dim, layer_units[0]))
            layers.append(nn.ReLU())

            for i in range(1, len(layer_units)):
                layers.append(nn.Linear(layer_units[i-1], layer_units[i]))
                layers.append(nn.ReLU())

            # Add the output layer
            layers.append(nn.Linear(layer_units[-1], self.output_dim))

            # Convert to a model
            model = nn.Sequential(*layers).to(self.device)

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            model.train()
            for epoch in range(13):
                self.logger.debug(f"NAS training epoch {epoch + 1}/13")
                for inputs, labels in train_loader:
                    self.logger.debug(f"Training input shape: {inputs.shape}, label shape: {labels.shape}")
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()

                    with amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(inputs)
                        self.logger.debug(f"Output shape: {outputs.shape}")
                        loss = criterion(outputs, labels)
                    
                    self.logger.debug(f"Loss: {loss.item()}")
                    
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

            # Evaluate the model on the validation data
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    self.logger.debug(f"Validation input shape: {inputs.shape}, label shape: {labels.shape}")
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    self.logger.debug(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                    val_loss += criterion(outputs, labels).item()

            self.logger.debug(f"Validation loss: {val_loss / len(val_loader)}")
            return val_loss / len(val_loader)
        
        except Exception as e:
            self.logger.error(f"Error during loss calculation: {e}")
            raise


    def refine(self, optimizer_params, meta_learner, reasoning_engine):
        try:
            self.logger.debug("Refining model with meta_learner and constraints")

            def black_box_function(*layer_units):
                self.logger.debug(f"Evaluating architecture with layer_units: {layer_units}")
                return -self.nas_algorithm(self.train_loader, self.val_loader, *layer_units)

            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds={'layer1_units': (64, 256), 'layer2_units': (64, 256), 'layer3_units': (32, 128)},
                random_state=1,
            )
            optimizer.maximize(init_points=10, n_iter=50)

            bayes_params = optimizer.max['params']
            self.logger.debug(f"Bayesian Optimization parameters found: {bayes_params}")

            # Instead of hardcoding layer units, make it dynamic
            best_layer_units = [int(bayes_params[f'layer{i}_units']) for i in range(1, 4)]  # Adjust for dynamic layers

            best_optimizer_params = self.search(self.training_data, self.validation_data)
            updated_architecture = self.architecture.copy()

            # Update the architecture based on the meta_learner layers and best parameters found
            for i, layer in enumerate(meta_learner.layers):
                updated_architecture[i] = layer.weight.data

            if 'non_negative' in reasoning_engine:
                updated_architecture[2] = torch.nn.functional.relu(updated_architecture[2])

            self.architecture = updated_architecture

            self.meta_learner.refine(updated_architecture, best_optimizer_params, reasoning_engine)

            query = 'Some query to the reasoning engine'
            results = self.reasoning_engine.infer(query)
            self.logger.debug(f"Reasoning engine results: {results}")

        except Exception as e:
            self.logger.error(f"Error during refinement: {e}")
            raise


    def get_state(self):
        state = {
            'architecture': [layer.state_dict() for layer in self.layers],
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }
        return state

    def set_state(self, state):
        self.architecture = [nn.Linear(state['input_dim'], state['architecture'][0]['weight'].shape[0]).to(self.device)]
        for layer_state in state['architecture']:
            layer = nn.Linear(layer_state['weight'].shape[1], layer_state['weight'].shape[0]).to(self.device)
            
            # Ensure the layer_state contains valid weight and bias parameters
            if 'weight' in layer_state:
                layer.load_state_dict(layer_state)
            else:
                raise ValueError(f"Missing 'weight' in layer_state: {layer_state}")
            
            self.architecture.append(layer)
        
        self.input_dim = state['input_dim']
        self.output_dim = state['output_dim']

    
    def propose_architecture(self, meta_learner):
        self.logger.debug("Proposing new architecture based on meta-learner")
        try:
            # Generate dummy data for training and validation
            dummy_input = torch.randn(100, self.input_dim).to(self.device)
            dummy_output = torch.randn(100, self.output_dim).to(self.device)
            dummy_train = TensorDataset(dummy_input, dummy_output)
            dummy_val = TensorDataset(dummy_input[:20], dummy_output[:20])
            
            train_loader = DataLoader(dummy_train, batch_size=32, shuffle=True)
            val_loader = DataLoader(dummy_val, batch_size=32, shuffle=False)
            
            # Run the search process to find the best architecture
            self.search(train_loader, val_loader)
            
            # Convert the architecture to a dictionary format
            architecture_dict = {}
            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Linear):
                    architecture_dict[f'layer_{i}'] = {
                        'type': 'Linear',
                        'in_features': layer.in_features,
                        'out_features': layer.out_features
                    }
                elif isinstance(layer, nn.ReLU):
                    architecture_dict[f'layer_{i}'] = {'type': 'ReLU'}
                elif isinstance(layer, nn.Dropout):
                    architecture_dict[f'layer_{i}'] = {'type': 'Dropout', 'p': layer.p}
                elif isinstance(layer, nn.BatchNorm1d):
                    architecture_dict[f'layer_{i}'] = {'type': 'BatchNorm1d', 'num_features': layer.num_features}
                elif isinstance(layer, nn.Conv2d):
                    architecture_dict[f'layer_{i}'] = {
                        'type': 'Conv2d',
                        'in_channels': layer.in_channels,
                        'out_channels': layer.out_channels,
                        'kernel_size': layer.kernel_size,
                        'stride': layer.stride,
                        'padding': layer.padding
                    }
                else:
                    # Log a warning for unsupported layer types
                    self.logger.warning(f"Unsupported layer type: {type(layer).__name__}")
            
            self.logger.debug(f"Proposed architecture: {architecture_dict}")
            return architecture_dict
        
        except Exception as e:
            self.logger.error(f"Error in proposing architecture: {e}")
            raise


    def diagnostics(self):
        diagnostics_info = {
            'num_layers': len(self.layers),
            'activation_functions': [str(layer) for layer in self.layers if isinstance(layer, nn.ReLU)],
            'total_params': sum(p.numel() for p in self.parameters())
        }
        return diagnostics_info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)



if __name__ == "__main__":
    logger.info("Starting main script for integrated NAS and EVO optimization")
    from AdvancedMetaLearner import AdvancedMetaLearner
    logger.info("Imported AdvancedMetaLearner")

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
    meta_learner = AdvancedMetaLearner(config=config)
    if torch.cuda.is_available():
        meta_learner.to('cuda')

    try:
        # Set up data and parameters
        input_dim = 10
        output_dim = 3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Generate dummy data
        train_size, val_size = 1000, 200
        logger.info(f"Generating dummy data: {train_size} training samples, {val_size} validation samples")
        training_data = (torch.randn(train_size, config['input_dim'], device=device), 
                         torch.randn(train_size,  config['output_dim'], device=device))
        validation_data = (torch.randn(val_size, config['input_dim'], device=device), 
                           torch.randn(val_size, config['output_dim'], device=device))
        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(*training_data), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(*validation_data), batch_size=32, shuffle=False)
        

        # Step 1: Neural Architecture Search
        logger.info("Starting Neural Architecture Search")
        nas = NeuralArchitectureSearch(config['input_dim'], config['output_dim'], device=device)
        nas.search(train_loader, val_loader)
        
        nas_architecture = nas.propose_architecture(meta_learner)
        logger.info("NAS completed. Proposed architecture: {meta_learner.layers}")
        logger.info(f"NAS completed. Proposed architecture: {nas_architecture} and the best architecture found: {nas.layers}")
        logger.info(f"NAS completed. Proposed architecture: {nas_architecture}")

        # Step 2: Evolutionary Optimization
        logger.info("Initializing EvolutionaryOptimizer with NAS architecture")
        evo_opt = EvolutionaryOptimizer(
            input_dim=config['input_dim'], 
            output_dim=config['output_dim'], 
            training_data=training_data, 
            validation_data=validation_data, 
            device=device,
            population_size=50,
            mutation_rate_start=0.5,
            mutation_rate_decay=0.95,
            mutation_rate=0.03,
            nas_architecture=nas_architecture
        )

        # Run the evolution
        num_generations = 50
        logger.info(f"Starting evolution for {num_generations} generations")
        evo_opt.evolve(num_generations, device)

        # After evolution, access the best individuals
        best_individuals = tools.selBest(evo_opt.population, k=5)
        for i, ind in enumerate(best_individuals):
            logger.info(f"Best individual {i+1} fitness: {ind.fitness.values}")

        # Save final elite weights
        torch.save([ind for ind in best_individuals], "checkpoints/final_elite_weights.pt")
        logger.info("Saved elite weights to checkpoints/final_elite_weights.pt")

        # Close the TensorBoard writer
        evo_opt.writer.close()

        logger.info("Integrated NAS and EVO optimization completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
        raise

    finally:
        logger.info("Script execution finished.")