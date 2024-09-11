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

    def refine(self, optimizer_params, meta_learner, reasoning_engine):
        try:
            self.logger.debug("Refining model with meta_learner and constraints")

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

            bayes_layer1_units = int(bayes_params['layer1_units'])
            bayes_layer2_units = int(bayes_params['layer2_units'])
            bayes_layer3_units = int(bayes_params['layer3_units'])

            initial_optimizer_params = [bayes_layer1_units, bayes_layer2_units, bayes_layer3_units]
            best_optimizer_params = self.evolutionary_optimizer.optimize(initial_optimizer_params)

            updated_architecture = self.architecture.copy()
            updated_architecture[0] = meta_learner.layers[0].weight.data
            updated_architecture[1] = meta_learner.layers[1].weight.data
            updated_architecture[2] = torch.tensor(best_optimizer_params, dtype=torch.float32, device=self.device)

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
            layer.load_state_dict(layer_state)
            self.architecture.append(layer)
        self.input_dim = state['input_dim']
        self.output_dim = state['output_dim']
    
    def propose_architecture(self, meta_learner):
        self.logger.debug("Proposing new architecture based on meta-learner")
        try:
            # Perform a search (this part remains the same)
            dummy_input = torch.randn(100, self.input_dim).to(self.device)
            dummy_output = torch.randn(100, self.output_dim).to(self.device)
            dummy_train = TensorDataset(dummy_input, dummy_output)
            dummy_val = TensorDataset(dummy_input[:20], dummy_output[:20])
            
            train_loader = DataLoader(dummy_train, batch_size=32, shuffle=True)
            val_loader = DataLoader(dummy_val, batch_size=32, shuffle=False)
            
            self.search(train_loader, val_loader)
            
            # Convert the best architecture to a dictionary format
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
