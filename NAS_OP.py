import os
import logging
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch import amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from multiprocessing import set_start_method
import tensorflow as tf
from deap import creator, base, tools
from bayes_opt import BayesianOptimization
from typing import Dict, List, Optional, Tuple, Any
import datetime
import time

# Set the start method to 'spawn' for multiprocessing
set_start_method('spawn', force=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the environment variable for CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Initialize TensorFlow to manage GPU resources
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        logger.error(f"GPU initialization error: {e}")

# Define the creator for the DEAP evolutionary algorithm
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Create directories if they don't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Contextual Logger for more specific logging
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

# Evolutionary Optimizer for population-based optimization
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
        self.toolbox.register("evaluate", self.evaluate_individual_gpu)  # Corrected line

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
        logger = logging.getLogger('EvolutionaryOptimizer')
        handler = logging.FileHandler('logs/evo_optimizer.log')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger

    async def optimize(self, parameters):
        try:
            self.logger.info("Starting optimization with Evolutionary Optimizer.")
            num_generations = 10  # You can adjust this value based on your needs
            for generation in range(num_generations):
                self.logger.info(f"Generation {generation + 1}/{num_generations}")
                offspring = await self.evolve_population()
                self.population = self.toolbox.select(self.population + offspring, self.population_size)

            best_individual = tools.selBest(self.population, 1)[0]
            self.logger.info("Optimization complete. Best individual selected.")

            optimized_proposal = self._convert_to_proposal(best_individual)
            return optimized_proposal

        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            raise
        
    def to(self, device: torch.device):
        self.device = device
        if hasattr(self, 'model'):
            self.model.to(device)
        self.logger.info(f"Moved EvolutionaryOptimizer to {device}")
        return self
    
    def initialize_individual_with_nas(self) -> Any:
        self.logger.debug("Initializing individual with NAS architecture")
        if self.nas_architecture is None:
            return self.initialize_individual()

        layers = []
        for layer_name, layer_info in self.nas_architecture.items():
            if isinstance(layer_info, dict) and 'in_features' in layer_info and 'out_features' in layer_info:
                layers.append(torch.randn(layer_info['out_features'], layer_info['in_features'], device=self.device))
            else:
                self.logger.warning(f"Unexpected layer info in NAS architecture: {layer_name}: {layer_info}")

        self.logger.debug(f"Individual initialized with NAS-based layer shapes: {[layer.shape for layer in layers]}")
        return creator.Individual(layers)

    def evaluate_population(self, population: List[Any]) -> List[Tuple[float, float]]:
        self.logger.debug("Evaluating population")
        fitnesses = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.build_model_from_nas(self.nas_architecture).to(device)
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        for ind in population:
            try:
                fit = self.evaluate_individual_gpu(ind, self.input_dim, self.output_dim, self.training_data, self.validation_data, device, self.nas_architecture, model)
                fitnesses.append(fit)
            except Exception as e:
                self.logger.error(f"Error during GPU evaluation: {e}")
                raise

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        return fitnesses

    def build_model_from_nas(self, nas_architecture):
        if nas_architecture is None:
            return nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_dim)
            )
        
        layers = []
        for layer_name, layer_info in nas_architecture.items():
            if layer_info['type'] == 'Linear':
                layers.append(nn.Linear(layer_info['in_features'], layer_info['out_features']))
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def initialize_population(self) -> List[Any]:
        self.logger.debug("Initializing population")
        try:
            population = [self.initialize_individual() for _ in range(self.population_size)]
            self.logger.debug(f"Population initialized with {len(population)} individuals")
            return population
        except Exception as e:
            self.logger.error(f"Error during population initialization: {e}")
            raise
        
    def initialize_individual(self) -> Any:
        self.logger.debug("Initializing individual")
        try:
            if self.nas_architecture:
                layers = []
                for i in range(len(self.nas_architecture) - 1):
                    in_features = self.nas_architecture[f'layer_{i}']['out_features']
                    out_features = self.nas_architecture[f'layer_{i+1}']['out_features']
                    layers.append(torch.randn(out_features, in_features, device=self.device))
            else:
                layers = [
                    torch.randn(256, self.input_dim, device=self.device),
                    torch.randn(128, 256, device=self.device),
                    torch.randn(64, 128, device=self.device),
                    torch.randn(self.output_dim, 64, device=self.device)
                ]
            self.logger.debug(f"Individual initialized with layer shapes: {[layer.shape for layer in layers]}")
            return creator.Individual(layers)
        except Exception as e:
            self.logger.error(f"Error initializing individual: {e}")
            raise

    def evaluate_individual_gpu(self, individual: List[torch.Tensor], 
                                input_dim: int, 
                                output_dim: int, 
                                training_data: Tuple[torch.Tensor, torch.Tensor], 
                                validation_data: Tuple[torch.Tensor, torch.Tensor], 
                                device: torch.device,
                                nas_architecture: Any) -> Tuple[float, float]:
        self.logger.debug("Evaluating individual on GPU")
        
        try:
            model = self.build_model_from_nas(nas_architecture).to(device)
            self._apply_weights_to_model(model, individual)
            
            train_loader = self._prepare_data_loader(training_data, batch_size=32)
            val_loader = self._prepare_data_loader(validation_data, batch_size=32)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            model.train()
            for epoch in range(5):
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
            
            fitness = total_loss / len(val_loader)
            complexity = sum(torch.norm(weight).item() for weight in individual)
            
            self.logger.debug(f"Individual evaluation complete. Fitness: {fitness}, Complexity: {complexity}")
            return fitness, complexity
        
        except Exception as e:
            self.logger.error(f"Error in GPU evaluation: {e}")
            raise

    def _apply_weights_to_model(self, model: nn.Module, individual: List[torch.Tensor]):
        for (name, param), weight in zip(model.named_parameters(), individual):
            if param.data.shape == weight.shape:
                param.data.copy_(weight)
            else:
                self.logger.warning(f"Shape mismatch for {name}: param {param.data.shape}, weight {weight.shape}")

    def _prepare_data_loader(self, data: Tuple[torch.Tensor, torch.Tensor], batch_size: int) -> DataLoader:
        inputs, targets = data
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def mutate(self, individual: Any, indpb: float = 0.1) -> Tuple[Any]:
        self.logger.debug("Mutating individual")
        for layer in individual:
            if torch.rand(1).item() < indpb:
                layer += torch.randn_like(layer) * 0.1
        self.logger.debug("Mutation complete")
        return individual,

    def _convert_to_proposal(self, best_individual):
        proposal = {"layer_{}".format(i + 1): layer.tolist() for i, layer in enumerate(best_individual)}
        return proposal

    def reproduce(self, ind1: Any, ind2: Any) -> Tuple[Any, Any]:
        self.logger.debug("Models are Mating")
        for layer1, layer2 in zip(ind1, ind2):
            if torch.rand(1).item() < 0.5:
                layer1, layer2 = layer2, layer1
        self.logger.debug("Mating complete")
        return ind1, ind2

    def evolve(self, num_generations: int):
        self.logger.info(f"Starting evolution for {num_generations} generations")
        for gen in range(num_generations):
            self.gen_handler.evolve(gen, num_generations)
            self.logger.info(f"Generation {gen} completed")
        self.logger.info("Evolution completed")

    async def evolve_population(self):
        try:
            offspring = self.toolbox.select(self.population, len(self.population))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if torch.rand(1).item() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if torch.rand(1).item() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = await self.evaluate_population(invalid_ind)

            self.mutation_rate *= self.mutation_rate_decay

            self.logger.info(f"Evolved population. New offspring: {len(offspring)}, Invalid individuals: {len(invalid_ind)}")
            return offspring

        except Exception as e:
            self.logger.error(f"Error during population evolution: {e}")
            raise
        
    def load_state_dict(self) -> dict:
        self.logger.debug("Getting optimizer state")
        return {
            'population': self.population,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'generation': self.gen_handler.current_generation
        }

    def save_state_dict(self, state: dict):
        self.logger.debug("Setting optimizer state")
        self.population = state['population']
        self.input_dim = state['input_dim']
        self.output_dim = state['output_dim']
        self.gen_handler.current_generation = state['generation']
        self.logger.debug("Optimizer state updated")

    async def propose_parameters(self, meta_learner):
        try:
            self.logger.info("Proposing parameters using Evolutionary Optimizer with NAS integration")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            self.evolve(num_generations=10)

            best_individual = tools.selBest(self.population, k=1)[0]

            proposed_parameters = {}
            for i, layer in enumerate(best_individual):
                proposed_parameters[f"layer_{i+1}"] = layer.to(self.device).detach().numpy()

            self.logger.info(f"Proposed parameters based on NAS architecture: {proposed_parameters.keys()}")

            await meta_learner.receive_proposal(proposed_parameters)
            self.logger.info("Parameters communicated to meta-learner successfully.")

            return proposed_parameters

        except Exception as e:
            self.logger.error(f"Error in proposing parameters: {e}")
            raise
        
    def update_architecture(self, nas_architecture):
        self.logger.debug(f"Updating architecture: {nas_architecture}")
        self.nas_architecture = nas_architecture
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
        self.logger.debug(f"Evolving generation {generation}")
        
        offspring = self.evolutionary_optimizer.toolbox.select(
            self.evolutionary_optimizer.population, 
            len(self.evolutionary_optimizer.population)
        )
        offspring = list(map(self.evolutionary_optimizer.toolbox.clone, offspring))
        self.logger.debug(f"Selected {len(offspring)} offspring for next generation")

        elite_size = int(0.1 * len(self.evolutionary_optimizer.population))
        elites = tools.selBest(self.evolutionary_optimizer.population, elite_size)
        self.logger.debug(f"Selected {len(elites)} elite individuals")

        if elites:
            top_elite = elites[0]
            top_elite_count = 2
            for _ in range(top_elite_count):
                offspring.append(self.evolutionary_optimizer.toolbox.clone(top_elite))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if torch.rand(1).item() < 0.5:
                self.evolutionary_optimizer.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        self.logger.debug("Crossover operations completed")

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.1:
                offspring.append(self.evolutionary_optimizer.toolbox.clone(offspring[i]))
        self.logger.debug("Twins possibility applied")

        mutation_rate = self.mutation_rate * (1 - generation / max_generations)
        for mutant in offspring:
            if random.random() < mutation_rate:
                self.evolutionary_optimizer.toolbox.mutate(mutant)
                del mutant.fitness.values
        self.logger.debug("Mutation operations completed")
        self.mutation_rate *= self.mutation_rate_decay

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.evolutionary_optimizer.evaluate_population(invalid_ind)
        self.logger.debug(f"Evaluated {len(invalid_ind)} invalid individuals")

        self.evolutionary_optimizer.population[:] = elites + offspring[:-elite_size]

        self.save_elites(elites, generation)

        best_fitness = min(ind.fitness.values[0] for ind in self.evolutionary_optimizer.population)
        self.evolutionary_optimizer.writer.add_scalar('Best Fitness', best_fitness, generation)
        self.logger.info(f"Generation {generation}: Best Fitness = {best_fitness}")

        self.current_generation += 1
        self.logger.debug(f"Generation {generation} evolution completed")

class NeuralArchitectureSearch(nn.Module):
    def __init__(self, input_dim, output_dim, device=None):
        super(NeuralArchitectureSearch, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = amp.GradScaler(device='cuda') 

        self.logger = ContextualLogger(self.setup_logging(), 'NeuralArchitectureSearch')
        self.log_health_check()

        self.layers = self.initialize_layers(input_dim, output_dim)

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

    def initialize_layers(self, input_dim, output_dim):
        self.logger.debug(f"Initializing architecture with input_dim: {input_dim}, output_dim: {output_dim}")
        layers = [
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        ]
        return nn.ModuleList(layers).to(self.device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def search(self, training_data, validation_data):
        self.logger.debug("Starting Neural Architecture Search")

        def black_box_function(layer1_units, layer2_units, layer3_units):
            self.logger.debug(f"Evaluating architecture with layer1_units: {layer1_units}, layer2_units: {layer2_units}, layer3_units: {layer3_units}")
            return -self.nas_algorithm(training_data, validation_data, int(layer1_units), int(layer2_units), int(layer3_units))

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds={'layer1_units': (64, 512), 'layer2_units': (64, 512), 'layer3_units': (32, 256)},
            random_state=1,
        )
        optimizer.maximize(init_points=10, n_iter=50)

        best_params = optimizer.max['params']
        self.logger.debug(f"Best parameters found: {best_params}")
        best_layer1_units = int(best_params['layer1_units'])
        best_layer2_units = int(best_params['layer2_units'])
        best_layer3_units = int(best_params['layer3_units'])
        self.layers = self.build_custom_model(self.input_dim, best_layer1_units, best_layer2_units, best_layer3_units, self.output_dim)

    def build_custom_model(self, input_dim, layer1_units, layer2_units, layer3_units, output_dim):
        self.logger.debug(f"Building custom model with layer1_units: {layer1_units}, layer2_units: {layer2_units}, layer3_units: {layer3_units}")
        model_layers = [
            nn.Linear(input_dim, layer1_units),
            nn.ReLU(),
            nn.Linear(layer1_units, layer2_units),
            nn.ReLU(),
            nn.Linear(layer2_units, layer3_units),
            nn.ReLU(),
            nn.Linear(layer3_units, output_dim)
        ]
        self.logger.debug(f"Built custom model layers: {model_layers}")
        return nn.ModuleList(model_layers).to(self.device)

    def generate(self, input_data):
        self.logger.debug(f"Generating output with input data length: {len(input_data)}")
        try:
            start_time = time.time()
            input_tensor = input_data.clone().detach().to(self.device).float()
            self.logger.debug(f"Input tensor shape: {input_tensor.shape}")
            output = self.forward(input_tensor)
            self.logger.debug(f"Output tensor shape: {output.shape}")
            end_time = time.time()
            self.logger.info(f"Generation completed in {end_time - start_time} seconds")
            return output.cpu().detach().numpy()
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise

    def nas_algorithm(self, train_loader, val_loader, layer1_units, layer2_units, layer3_units):
        self.logger.debug(f"Running NAS algorithm with layer1_units: {layer1_units}, layer2_units: {layer2_units}, layer3_units: {layer3_units}")
        try:
            model = nn.Sequential(
                nn.Linear(self.input_dim, layer1_units),
                nn.ReLU(),
                nn.Linear(layer1_units, layer2_units),
                nn.ReLU(),
                nn.Linear(layer2_units, layer3_units),
                nn.ReLU(),
                nn.Linear(layer3_units, self.output_dim)
            ).to(self.device)

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
                    with amp.autocast(device_type='cuda'):
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

    def refine(self, optimizer_params, meta_learner, reasoning_engine, training_data=None, validation_data=None):
        try:
            self.logger.debug("Refining model with meta_learner and constraints")

            # Step 1: Perform Neural Architecture Search (NAS) using Bayesian Optimization
            def black_box_function(layer1_units, layer2_units, layer3_units):
                self.logger.debug(f"Evaluating architecture with layer1_units: {layer1_units}, layer2_units: {layer2_units}, layer3_units: {layer3_units}")
                return -self.nas_algorithm(self.train_loader, self.val_loader, int(layer1_units), int(layer2_units), int(layer3_units))

            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds={'layer1_units': (64, 512), 'layer2_units': (64, 512), 'layer3_units': (32, 256)},
                random_state=1,
            )
            optimizer.maximize(init_points=10, n_iter=50)

            bayes_params = optimizer.max['params']
            self.logger.debug(f"Bayesian Optimization parameters found: {bayes_params}")

            # Convert the best Bayesian parameters to initial evolutionary optimizer parameters
            bayes_layer1_units = int(bayes_params['layer1_units'])
            bayes_layer2_units = int(bayes_params['layer2_units'])
            bayes_layer3_units = int(bayes_params['layer3_units'])

            initial_optimizer_params = [bayes_layer1_units, bayes_layer2_units, bayes_layer3_units]

            # Step 2: Refine the architecture further using the evolutionary algorithm
            best_optimizer_params = self.search(self.training_data, self.validation_data)

            updated_architecture = self.architecture.copy()
            updated_architecture[0] = meta_learner.layers[0].weight.data
            updated_architecture[1] = meta_learner.layers[1].weight.data
            updated_architecture[2] = torch.tensor(best_optimizer_params, dtype=torch.float32, device=self.device)

            # Apply additional constraints using the reasoning engine
            if 'non_negative' in reasoning_engine:
                updated_architecture[2] = torch.nn.functional.relu(updated_architecture[2])

            self.architecture = updated_architecture

            # Step 3: Further refinement of the architecture using the meta-learner
            self.meta_learner.refine(updated_architecture, best_optimizer_params, reasoning_engine)

            # Query the reasoning engine with specific questions or scenarios
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
            layer.load_state_dict(layer_state)
            self.architecture.append(layer)
        self.input_dim = state['input_dim']
        self.output_dim = state['output_dim']
    
    def propose_architecture(self, meta_learner):
        self.logger.debug("Proposing new architecture based on meta-learner")
        try:
            dummy_input = torch.randn(100, self.input_dim).to(self.device)
            dummy_output = torch.randn(100, self.output_dim).to(self.device)
            dummy_train = TensorDataset(dummy_input, dummy_output)
            dummy_val = TensorDataset(dummy_input[:20], dummy_output[:20])
            
            train_loader = DataLoader(dummy_train, batch_size=32, shuffle=True)
            val_loader = DataLoader(dummy_val, batch_size=32, shuffle=False)
            
            self.search(train_loader, val_loader)
            
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

if __name__ == "__main__":
    logger.info("Starting main script")
    
    try:
        input_dim = 10
        output_dim = 3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        train_size, val_size = 1000, 200
        logger.info(f"Generating dummy data: {train_size} training samples, {val_size} validation samples")
        training_data = (torch.randn(train_size, input_dim, device=device), 
                         torch.randn(train_size, output_dim, device=device))
        validation_data = (torch.randn(val_size, input_dim, device=device), 
                           torch.randn(val_size, output_dim, device=device))

        logger.info("Initializing EvolutionaryOptimizer")
        evo_opt = EvolutionaryOptimizer(
            input_dim=input_dim, 
            output_dim=output_dim, 
            training_data=training_data, 
            validation_data=validation_data, 
            device=device,
            population_size=50,
            mutation_rate_start=0.5,
            mutation_rate_decay=0.95,
            mutation_rate=0.03
        )

        num_generations = 50
        logger.info(f"Starting evolution for {num_generations} generations")
        evo_opt.evolve(num_generations)

        best_individuals = tools.selBest(evo_opt.population, k=5)
        for i, ind in enumerate(best_individuals):
            logger.info(f"Best individual {i+1} fitness: {ind.fitness.values}")

        torch.save([ind for ind in best_individuals], "checkpoints/final_elite_weights.pt")

        evo_opt.writer.close()

        logger.info("Evolution completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise
