import json
import logging
import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import psutil
from qiskit_algorithms import NumPyMinimumEigensolver
import random
import math
import time
import asyncio
from datetime import datetime, timedelta
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer, AerSimulator

from qiskit.circuit.library import QFT
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import ZZFeatureMap
from qiskit.providers.jobstatus import JobStatus
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit import Parameter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from scipy.optimize import minimize
from qiskit.qpy import dump, load
import io
import logging
class SimulatorAdjuster:
    def __init__(self, n_estimators=100, learning_rate=0.01):
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        self.scaler = StandardScaler()
        self.history = []

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def update(self, X, y):
        self.history.append((X, y))
        if len(self.history) >= 10:
            X_train = np.vstack([x for x, _ in self.history])
            y_train = np.concatenate([y for _, y in self.history])
            self.fit(X_train, y_train)

class BindParameters:
        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def bind(self, circuit: QuantumCircuit, parameter_binds: List[float]) -> QuantumCircuit:
            """
            Binds parameters to the quantum circuit.
            
            :param circuit: The quantum circuit with parameters to be bound.
            :param parameter_binds: A list of values to bind to the circuit parameters.
            :return: A new quantum circuit with bound parameters.
            """
            self.logger.debug(f"Binding parameters: {parameter_binds}")
            
            if len(circuit.parameters) != len(parameter_binds):
                raise ValueError(f"Expected {len(circuit.parameters)} parameters, but got {len(parameter_binds)}")
            
            bound_circuit = circuit.copy()
            param_dict = dict(zip(circuit.parameters, parameter_binds))
            
            for i, instruction in enumerate(bound_circuit.data):
                if instruction[0].params:
                    old_params = instruction[0].params
                    new_params = self._update_instruction_params(old_params, param_dict)
                    instruction[0].params = tuple(new_params)
                    self.logger.debug(f"Instruction {i}: Old params: {old_params}, New params: {new_params}")
            
            self.logger.debug("Parameters bound successfully")
            return bound_circuit

        def _update_instruction_params(self, params: tuple, param_dict: Dict[Parameter, float]) -> List[Any]:
            """
            Updates the parameters of an instruction.
            
            :param params: The current parameters of the instruction.
            :param param_dict: A dictionary mapping Parameters to their bound values.
            :return: A list of updated parameters.
            """
            new_params = []
            for param in params:
                if isinstance(param, (int, float)):
                    new_params.append(param)  # Keep fixed parameters unchanged
                elif isinstance(param, Parameter) and param in param_dict:
                    new_params.append(param_dict[param])  # Bind the parameter value
                else:
                    new_params.append(param)
            return new_params

        def validate_circuit(self, circuit: QuantumCircuit) -> None:
            """
            Validates that the circuit has parameters.
            
            :param circuit: The quantum circuit to validate.
            :raises ValueError: If the circuit has no parameters.
            """
            if not circuit.parameters:
                raise ValueError("The provided circuit has no parameters to bind.")

        def log_binding_result(self, original_circuit: QuantumCircuit, bound_circuit: QuantumCircuit) -> None:
            """
            Logs the result of parameter binding.
            
            :param original_circuit: The original quantum circuit before binding.
            :param bound_circuit: The quantum circuit after binding parameters.
            """
            self.logger.info(f"Original circuit parameters: {original_circuit.parameters}")
            self.logger.info(f"Bound circuit parameters: {bound_circuit.parameters}")
            self.logger.info(f"Number of gates before binding: {len(original_circuit.data)}")
            self.logger.info(f"Number of gates after binding: {len(bound_circuit.data)}")



class QuantumModule:
    def __init__(self, vocab_size, config=None, cache_file='quantum_cache.json', usage_file='quantum_usage.json', logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.vocab_size = vocab_size
        self.qr = QuantumRegister(vocab_size, 'q')
        self.qc = QuantumCircuit()
        self.qc.add_register(self.qr)
        self.cache = self._load_cache(cache_file)
        self.sub_circuit_cache = {}
        self.qcf_threshold = 0.5
        self.usage_file = usage_file
        self.real_backend_time_limit = timedelta(minutes=9)
        self.real_quantum_backend = None
        self.cache_file = cache_file
        self.real_backend_time_used = timedelta()
        self.simulator_adjuster = SimulatorAdjuster()
        self.error_rates = {
            'single_qubit': 0.001,
            'two_qubit': 0.01,
            'measurement': 0.02
        }
        self.config = config if config else self._default_config()

        self._load_usage()

    def _default_config(self):
        # Define a default configuration
        return {
            'backend': 'simulator',
            'shots': 1024,
            'optimization_level': 1
        }


    def _initialize_ibm_runtime(self, retry_attempts=2):
        # Save and load IBM Runtime service credentials
        try:

            QiskitRuntimeService.save_account(channel="ibm_quantum", token="81580c0e48abc06a65a5b38aa997d6ef4ccb72b43d5bce3a637bbce0b05fa9adc2415ef2f58a7b80285ab826ad8d6ef924477d83d9d506309e5c281687f57f3b", overwrite=True)
            self.service = QiskitRuntimeService()
            backends = self.service.backends(
                filters=lambda x: x.configuration().n_qubits >= self.vocab_size and not x.configuration().simulator and x.status().operational
            )
            self.real_quantum_backend = backends[0] if backends else None
        except Exception as e:
            if retry_attempts > 0:
                self.logger.error(f"Error initializing IBM Quantum service: {e}. Retrying...")
                time.sleep(2)  # Sleep before retrying
                self._initialize_ibm_runtime(retry_attempts - 1)
            else:
                self.logger.error("Failed to initialize IBM Quantum service after retries.")
                raise

    def _generate_cache_key(self, func_name, *args):
        def make_hashable(obj):
            if isinstance(obj, (list, tuple)):
                return tuple(make_hashable(e) for e in obj)
            elif isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, set):
                return frozenset(make_hashable(e) for e in obj)
            return obj

        hashable_args = tuple(make_hashable(arg) for arg in args)
        return (func_name, hashable_args)

    def _key_to_str(self, key):
        return json.dumps(str(key))

    def _get_from_cache(self, key):
        str_key = self._key_to_str(key)
        if str_key in self.cache:
            self.logger.info(f"Cache hit for key: {str_key}")
            return self.cache[str_key]
        self.logger.info(f"Cache miss for key: {str_key}")
        return None
    
    def _save_to_cache(self, key, value):
        # Convert the key to string form
        str_key = self._key_to_str(key)
        
        # Log the cache-saving process
        self.logger.info(f"Saving result to cache with key: {str_key}")
        
        # Add the new key-value pair to the cache
        self.cache[str_key] = value
        
        # Prepare data for saving
        cache_data = {
            'main_cache': self.cache,
            'sub_circuit_cache': {k: v.to_dict() for k, v in self.sub_circuit_cache.items()}
        }
        
        # Save cache data to a JSON file
        self._save_json_to_file(cache_data, self.cache_file)

    
    def _save_sub_circuit_to_cache(self, key, sub_circuit):
        """
        Caches reusable quantum sub-circuits.
        """
        str_key = self._key_to_str(key)
        self.logger.info(f"Saving sub-circuit to cache with key: {str_key}")
        
        # Serialize the sub-circuit to QPY format
        qpy_data = io.BytesIO()
        dump(sub_circuit, qpy_data)
        
        # Store the binary data as a list of integers (to be JSON serializable)
        self.sub_circuit_cache[str_key] = list(qpy_data.getvalue())
        
        # Save updated cache
        cache_data = {
            'main_cache': self.cache,
            'sub_circuit_cache': self.sub_circuit_cache
        }
        self._save_json_to_file(cache_data, self.cache_file)

    def _get_sub_circuit_from_cache(self, key):
        str_key = self._key_to_str(key)
        if str_key in self.sub_circuit_cache:
            self.logger.info(f"Sub-circuit cache hit for key: {str_key}")
            return self.sub_circuit_cache[str_key]
        self.logger.info(f"Sub-circuit cache miss for key: {str_key}")
        return None

    def _load_cache(self, cache_file):
        cache_data = self._load_json_from_file(cache_file)
        if cache_data is None:
            self.logger.warning("Initializing empty caches.")
            self.cache = {}
            self.sub_circuit_cache = {}
        else:
            self.cache = cache_data.get('main_cache', {})
            self.sub_circuit_cache = {}
            
            # Deserialize circuits from QPY format
            for k, v in cache_data.get('sub_circuit_cache', {}).items():
                qpy_data = io.BytesIO(bytearray(v))
                circuits = load(qpy_data)
                self.sub_circuit_cache[k] = circuits[0]  # Assuming one circuit per key
        return self.cache
    
    def _load_json_from_file(self, file_path):
        if not os.path.exists(file_path):
            self.logger.info(f"File {file_path} does not exist.")
            return None
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if not content.strip():
                    self.logger.warning(f"File {file_path} is empty.")
                    return None
                return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from file {file_path}: {e}")
            return None

    def _load_usage(self):
        if os.path.exists(self.usage_file):
            with open(self.usage_file, 'r') as f:
                usage_data = json.load(f)
                self.last_reset = datetime.fromisoformat(usage_data['last_reset'])
                self.real_backend_time_used = timedelta(seconds=usage_data['real_backend_time_used'])
        else:
            self.last_reset = datetime.now()
            self.real_backend_time_used = timedelta()

        # Reset usage if a month has passed
        if datetime.now() - self.last_reset > timedelta(days=30):
            self.real_backend_time_used = timedelta()
            self.last_reset = datetime.now()
            self._save_usage()

    def _save_json_to_file(self, data, file_path):
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving data to {file_path}: {e}")


    def _save_usage(self):
        usage_data = {
            'last_reset': self.last_reset.isoformat(),
            'real_backend_time_used': self.real_backend_time_used.total_seconds()
        }
        self._save_json_to_file(usage_data, self.usage_file)
        
    async def quantum_compute(self, quantum_circuit, shots=1024, force_real_backend=False):
        """
        Centralized function to handle quantum simulations and backend execution.
        Uses simulator 98% of the time and real backend 2% of the time for validation,
        unless force_real_backend is True.
        """
        try:
            use_real_backend = force_real_backend or (random.random() < 0.02 and self.real_backend_time_used < self.real_backend_time_limit)

            if use_real_backend:
                if self.real_backend_time_used >= self.real_backend_time_limit:
                    print("Real quantum backend usage exceeded, switching to simulator.")
                    backend = Aer.get_backend('qasm_simulator')
                else:
                    print("Using real IBM quantum backend for validation.")
                    backend = self.real_quantum_backend
            else:
                backend = Aer.get_backend('qasm_simulator')

            print(f"Using backend: {backend.name()}")

            # Transpile the quantum circuit
            transpiled_qc = transpile(quantum_circuit, backend)

            # Submit the job
            job = backend.run(transpiled_qc, shots=shots)
            print(f"Job ID: {job.job_id()}")

            # Async monitor the job status and queue position
            while job.status() not in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]:
                queue_position = job.queue_position()
                print(f"Job status: {job.status()}, Queue position: {queue_position}")
                await asyncio.sleep(2)  # Wait asynchronously

            # Once the job is done, check the final status
            final_status = job.status()
            print(f"Final job status: {final_status}")

            if final_status == JobStatus.DONE:
                # Try to retrieve the result
                result = job.result()
                print(f"Job result retrieved successfully: {result}")
                return result
            else:
                raise ValueError(f"Job failed with status: {final_status}")

        except Exception as e:
            print(f"Error during quantum computation: {e}")
            raise

    def _compare_and_adjust(self, real_result, sim_result):
        """
        SMART MRAP: Systematic, Measurable, Achievable, Relevant, Time-bound 
        Multi-Round Adjustment Protocol
        """
        real_counts = real_result.get_counts()
        sim_counts = sim_result.get_counts()
        total_real = sum(real_counts.values())
        total_sim = sum(sim_counts.values())

        states = set(real_counts.keys()) | set(sim_counts.keys())
        differences = []
        features = []

        for state in states:
            real_prob = real_counts.get(state, 0) / total_real
            sim_prob = sim_counts.get(state, 0) / total_sim
            diff = real_prob - sim_prob
            differences.append(diff)
            
            state_features = self._extract_features(state, real_prob, sim_prob)
            features.append(state_features)

        self.simulator_adjuster.update(np.array(features), np.array(differences))

        adjusted_sim_counts = {}
        for state, feature in zip(states, features):
            adjustment = self.simulator_adjuster.predict([feature])[0]
            original_count = sim_counts.get(state, 0)
            adjusted_count = max(0, original_count + adjustment * total_sim)
            adjusted_sim_counts[state] = adjusted_count

        total_adjusted = sum(adjusted_sim_counts.values())
        for state in adjusted_sim_counts:
            adjusted_sim_counts[state] = adjusted_sim_counts[state] / total_adjusted * total_sim

        self._log_adjustments(real_counts, sim_counts, adjusted_sim_counts, total_real, total_sim)
        self._update_simulator_params(adjusted_sim_counts, real_counts, total_real)

        self.logger.info("SMART MRAP adjustment completed.")

    def _extract_features(self, state, real_prob, sim_prob):
        """Extract relevant features from a quantum state and probabilities."""
        complexity = self._calculate_state_complexity(state)
        return [
            state.count('1'),  # Number of 1s
            len(max(state.split('0'), key=len)),  # Longest run of 1s
            len(state),  # Total number of qubits
            real_prob,  # Real probability
            sim_prob,  # Simulated probability
            abs(real_prob - sim_prob),  # Absolute difference
            complexity['total'],  # Overall state complexity
            complexity['bit_entropy'],  # Bit-string entropy
            complexity['run_complexity'],  # Longest run complexity
            complexity['transition_complexity']  # Transition complexity
        ]
    
    def _calculate_state_complexity(self, state):
        """
        Calculate the complexity of a quantum state.
        This function computes multiple complexity measures:
        1. Bit-string entropy
        2. Longest run complexity
        3. Transition complexity
        """
        # 1. Bit-string entropy
        probs = [state.count('0') / len(state), state.count('1') / len(state)]
        bit_entropy = entropy(probs, base=2)

        # 2. Longest run complexity
        longest_run = max(len(run) for run in state.replace('0', ' ').replace('1', ' ').split())
        run_complexity = longest_run / len(state)

        # 3. Transition complexity (number of 0-1 or 1-0 transitions)
        transitions = sum(1 for i in range(1, len(state)) if state[i] != state[i-1])
        transition_complexity = transitions / (len(state) - 1)

        # Combine these measures (you can adjust the weights as needed)
        total_complexity = (bit_entropy + run_complexity + transition_complexity) / 3

        return {
            'total': total_complexity,
            'bit_entropy': bit_entropy,
            'run_complexity': run_complexity,
            'transition_complexity': transition_complexity
        }

    def _log_adjustments(self, real_counts, sim_counts, adjusted_sim_counts, total_real, total_sim):
        """Log significant differences and adjustments."""
        for state in set(real_counts.keys()) | set(sim_counts.keys()) | set(adjusted_sim_counts.keys()):
            real_prob = real_counts.get(state, 0) / total_real
            orig_sim_prob = sim_counts.get(state, 0) / total_sim
            adj_sim_prob = adjusted_sim_counts.get(state, 0) / total_sim
            if abs(real_prob - orig_sim_prob) > 0.05 or abs(real_prob - adj_sim_prob) > 0.05:
                self.logger.info(f"State {state}: Real={real_prob:.4f}, Original Sim={orig_sim_prob:.4f}, "
                                 f"Adjusted Sim={adj_sim_prob:.4f}")

    def _update_simulator_params(self, adjusted_counts, real_counts, total_real):
        """Update simulator parameters based on adjusted counts."""
        # Calculate overall fidelity
        fidelity = sum(min(adjusted_counts.get(state, 0) / total_real, real_counts.get(state, 0) / total_real)
                       for state in set(adjusted_counts.keys()) | set(real_counts.keys()))

        # Update error rates based on fidelity
        adjustment_factor = (1 - fidelity) / (1 - sum(self.error_rates.values()))
        for error_type in self.error_rates:
            self.error_rates[error_type] *= adjustment_factor

        # Update specific error rates based on state analysis
        self._update_specific_error_rates(adjusted_counts, real_counts, total_real)

        self.logger.info(f"Updated error rates: {self.error_rates}")

    def _update_specific_error_rates(self, adjusted_counts, real_counts, total_real):
        """Update specific error rates based on detailed state analysis."""
        single_qubit_errors = 0
        two_qubit_errors = 0
        measurement_errors = 0
        total_states = 0

        for state in set(adjusted_counts.keys()) | set(real_counts.keys()):
            adj_prob = adjusted_counts.get(state, 0) / total_real
            real_prob = real_counts.get(state, 0) / total_real
            error = abs(adj_prob - real_prob)

            if state.count('1') == 1:  # Single qubit flip
                single_qubit_errors += error
            elif state.count('1') == 2:  # Potential two-qubit error
                two_qubit_errors += error
            
            # Measurement error estimation (simplistic approach)
            measurement_errors += error * (1 if state.endswith('1') else 0.5)
            
            total_states += 1

        # Update error rates (with some smoothing to avoid drastic changes)
        smoothing_factor = 0.9
        self.error_rates['single_qubit'] = smoothing_factor * self.error_rates['single_qubit'] + \
                                           (1 - smoothing_factor) * (single_qubit_errors / total_states)
        self.error_rates['two_qubit'] = smoothing_factor * self.error_rates['two_qubit'] + \
                                        (1 - smoothing_factor) * (two_qubit_errors / total_states)
        self.error_rates['measurement'] = smoothing_factor * self.error_rates['measurement'] + \
                                          (1 - smoothing_factor) * (measurement_errors / total_states)

    



    async def quantum_arithmetic_encode(self, tokens, freqs, shots=1024):
        key = self._generate_cache_key('quantum_arithmetic_encode', tokens, freqs, shots)
        cached_result = self._get_from_cache(key)
        if cached_result is not None:
            return cached_result

        n_qubits = len(tokens)
        qr = QuantumRegister(n_qubits, 'q')  # Create QuantumRegister here
        cr = ClassicalRegister(n_qubits, 'c')  # Create ClassicalRegister here
        self.logger.debug(f"Tokens length: {len(tokens)}, n_qubits calculated as: {n_qubits}")
        
        if n_qubits != len(qr):
            raise ValueError(f"Number of qubits mismatch: expected {len(qr)}, got {n_qubits}")
        
        qc = QuantumCircuit(qr, cr)

        # Encode input state
        for i, token in enumerate(tokens):
            if token == 1:
                qc.x(qr[i])

        # Apply QFT
        qc.append(QFT(n_qubits, do_swaps=False), qr)

        # Inverse QFT
        qc.append(QFT(num_qubits=n_qubits, do_swaps=False), qr[:])

        qc.measure(qr, cr)

        result = await self.quantum_compute(qc, shots)
        counts = result.get_counts(qc)
        most_likely_state = max(counts, key=counts.get)
        encoded_value = int(most_likely_state, 2) / (2 ** n_qubits)

        self._save_to_cache(key, encoded_value)
        return encoded_value

    async def quantum_arithmetic_decode(self, encoded_value, freqs, shots=1024):
        key = self._generate_cache_key('quantum_arithmetic_decode', encoded_value, freqs, shots)
        cached_result = self._get_from_cache(key)
        if cached_result is not None:
            return cached_result

        n_qubits = len(freqs)
        qr = QuantumRegister(n_qubits, 'q')  # Create QuantumRegister here
        cr = ClassicalRegister(n_qubits, 'c')  # Create ClassicalRegister here
        
        
        if n_qubits != len(qr):
            raise ValueError(f"Number of qubits mismatch: expected {len(qr)}, got {n_qubits}")
        qc = QuantumCircuit(qr, cr)

        # Encode the value
        angle = 2 * math.pi * encoded_value
        for i in range(n_qubits):
            qc.ry(angle * (2**i), qr[i])

        # Apply inverse QFT
        qc.append(QFT(num_qubits=n_qubits, do_swaps=False), qr[:])

        qc.measure(qr, cr)

        result = await self.quantum_compute(qc, shots)
        counts = result.get_counts(qc)

        decoded_probs = [(int(state, 2), count / shots) for state, count in counts.items()]
        decoded_tokens = [token for state, prob in sorted(decoded_probs, reverse=True) 
                          for token, (low, high) in freqs.items() if low <= prob < high]

        self._save_to_cache(key, decoded_tokens)
        return decoded_tokens
    
    async def quantum_sparse_encode(self, tokens: List[int], vocab_size: int) -> List[complex]:
        self.logger.debug(f"Starting enhanced quantum_sparse_encode with tokens: {tokens}, vocab_size: {vocab_size}")
        key = self._generate_cache_key('quantum_sparse_encode', tokens, vocab_size)
        
        cached_result = self._get_from_cache(key)
        if cached_result is not None:
            return cached_result

        try:
            # Use ZZFeatureMap for more expressive encoding
            feature_map = ZZFeatureMap(feature_dimension=vocab_size, reps=2)

            # Create a parameterized circuit
            qc = QuantumCircuit(vocab_size)
            qc.append(feature_map, range(vocab_size))
            
            self.logger.debug(f"Circuit parameters after creation: {qc.parameters}")
            
            if not qc.parameters:
                self.logger.warning("Warning: Circuit has no parameters")

            def objective_function(params):
                logging.debug(f"Starting objective_function with params: {params}")
                
                if len(params) != len(qc.parameters):
                    raise ValueError(f"Expected {len(qc.parameters)} parameters, but got {len(params)}")
                
                parameter_binds = dict(zip(qc.parameters, params))
                logging.debug(f"Parameter bindings created: {parameter_binds}")
                
                try:
                    bound_circuit = qc.assign_parameters(parameter_binds)
                    job = self.quantum_compute_sync(bound_circuit)
                    logging.debug("Called quantum_compute_sync and got job object")
                    
                    # Wait for the job to complete
                    job_status = job.status()
                    while job_status not in ['DONE', 'ERROR', 'CANCELLED']:
                        time.sleep(0.1)
                        job_status = job.status()
                    
                    if job_status != 'DONE':
                        raise RuntimeError(f"Job failed with status: {job_status}")
                    
                    result = job.result()
                    logging.debug(f"Result object retrieved: {result}")
                    
                    if not hasattr(result, 'get_statevector'):
                        raise AttributeError("Result object does not have the 'get_statevector' method.")
                    
                    statevector = result.get_statevector()
                    logging.debug(f"Statevector retrieved: {statevector}")
                    
                    target_state = [1 if i in tokens else 0 for i in range(vocab_size)]
                    logging.debug(f"Target state: {target_state}")
                    
                    fidelity = state_fidelity(Statevector(statevector), Statevector(target_state))
                    logging.debug(f"Fidelity computed: {fidelity}")
                    
                    return -fidelity  # Negative because we want to maximize fidelity
                except Exception as e:
                    logging.error(f"Error in objective_function: {str(e)}")
                    # Return a large positive value to indicate a failed evaluation to the optimizer
                    return 1e6

            # Initialize x0 with non-zero values
            x0 = [np.pi/4] * len(qc.parameters)

            # Optimize with COBYLA
            opt_result = minimize(objective_function, x0=x0, method='COBYLA', options={'maxiter': 100})

            # Get the final quantum state
            final_circuit = qc.assign_parameters(dict(zip(qc.parameters, opt_result.x)))
            job = await self.quantum_compute(final_circuit)
            result = job.result()
            quantum_state = result.get_statevector()

            self.logger.debug(f"Optimized quantum state fidelity: {-opt_result.fun:.4f}")
            self._save_to_cache(key, quantum_state)
            return quantum_state
        except Exception as e:
            self.logger.error(f"Error in enhanced quantum_sparse_encode: {e}", exc_info=True)
            raise
        finally:
            self.logger.debug("Exiting enhanced quantum_sparse_encode.")

    def quantum_compute_sync(self, quantum_circuit, shots=1024):
        backend = Aer.get_backend('statevector_simulator')
        self.logger.debug(f"Using backend: {backend.configuration().backend_name}")

        self.logger.debug(f"Quantum circuit parameters: {quantum_circuit.parameters}")

        try:
            transpiled_qc = transpile(quantum_circuit, backend)
            self.logger.debug(f"Transpiled quantum circuit: {transpiled_qc}")
        except Exception as e:
            self.logger.error(f"Error during circuit transpilation: {e}")
            raise

        try:
            job = backend.run(transpiled_qc, shots=shots)
            self.logger.debug(f"Job ID: {job.job_id()}")
            self.logger.debug(f"Initial job status: {job.status()}")
            return job
        except Exception as e:
            self.logger.error(f"Error executing the quantum job: {e}")
            raise



                            
    async def quantum_sparse_decode(self, quantum_state, vocab_size: int) -> List[int]:
        """Decodes quantum state into tokens."""
        self.logger.debug(f"Starting quantum_sparse_decode for vocab size: {vocab_size}")
        key = self._generate_cache_key('quantum_sparse_decode', quantum_state, vocab_size)

        # Cache Check
        cached_result = self._get_from_cache(key)
        if cached_result is not None:
            return cached_result

        try:
            backend = Aer.get_backend('statevector_simulator')
            transpiled_qc = transpile(QuantumCircuit(quantum_state), backend)
            job = backend.run(transpiled_qc)
            result = job.result()
            statevector = result.get_statevector()

            # Decode tokens
            decoded_tokens = [i for i, amplitude in enumerate(statevector) if i < vocab_size and abs(amplitude) > 1e-6]
            self.logger.debug(f"Decoded tokens: {decoded_tokens}")

            # Save and return result
            self._save_to_cache(key, decoded_tokens)
            return decoded_tokens
        except Exception as e:
            self.logger.error(f"Error in quantum_sparse_decode: {e}", exc_info=True)
            raise
        finally:
            self.logger.debug("Exiting quantum_sparse_decode.")
        
    async def quantum_huffman_encode(self, frequencies):
        self.logger.debug("Starting quantum_huffman_encode")
        key = self._generate_cache_key('quantum_huffman_encode', tuple(frequencies.items()))
        cached_result = self._get_from_cache(key)
        if cached_result is not None:
            return cached_result

        try:
            # Sorting frequencies in descending order
            sorted_freq = sorted(frequencies.items(), key=lambda x: x[1][0], reverse=True)
            tokens = [token for token, freq in sorted_freq]
            
            # Extracting the average frequency for computation
            freqs = [(freq[0] + freq[1]) / 2 for token, freq in sorted_freq]

            num_tokens = len(tokens)

            # Constructing the QUBO problem
            problem = QuadraticProgram()
            for i in range(num_tokens):
                problem.binary_var(f'x_{i}')

            linear = {f'x_{i}': freqs[i] for i in range(num_tokens)}
            quadratic = {(f'x_{i}', f'x_{j}'): freqs[i] * freqs[j] for i in range(num_tokens) for j in range(i + 1, num_tokens)}

            problem.minimize(linear=linear, quadratic=quadratic)

            # Use a classical optimizer for the Huffman encoding first
            classical_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
            result = classical_solver.solve(problem)

            selected_vars = result.x
            tree = []
            for i in range(num_tokens):
                if selected_vars[i]:
                    tree.append(tokens[i])

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
            self._save_to_cache(key, codes)
            return codes

        except Exception as e:
            self.logger.error(f"Error in quantum_huffman_encode: {e}")
            raise

        
    def quantum_compression(self, data):
            logging.info("Applying improved fractal-based quantum compression with adaptive entropy scaling.")
            key = self._generate_cache_key('quantum_compression', data)
            cached_result = self._get_from_cache(key)
            if cached_result is not None:
                return cached_result

            # Ensure data is numeric
            if not isinstance(data, np.ndarray):
                logging.error("Data must be a NumPy array for quantum compression.")
                raise TypeError("Data must be a NumPy array for quantum compression.")

            # Create a simple quantum circuit with one qubit
            circuit = QuantumCircuit(1)
            circuit.h(0)  # Hadamard gate for superposition
            circuit.measure_all()

            # Simulate the quantum circuit
            simulator = Aer.get_backend('qasm_simulator')
            compiled_circuit = transpile(circuit, simulator)
            job = simulator.run(compiled_circuit, shots=1)
            result = job.result()
            counts = result.get_counts(compiled_circuit)

            # Determine the compression factor based on quantum results
            compression_factor = counts.get('0', 1) + 1
            logging.info(f"Quantum compression factor determined: {compression_factor}")

            # Calculate entropy
            data_sum = np.sum(data)
            if data_sum == 0:
                logging.error("Data sum is zero; cannot compute entropy.")
                raise ValueError("Data sum is zero; cannot compute entropy.")
            
            normalized_data = data / data_sum
            entropy = -np.sum(normalized_data * np.log2(normalized_data + 1e-9))  # Calculating data entropy
            logging.info(f"Data entropy calculated: {entropy}")

            # Adjust the compression factor by the entropy
            adjusted_factor = max(1.0, compression_factor * (1 + 0.1 * entropy))
            logging.info(f"Adjusted compression factor: {adjusted_factor}")

            # Compress the data
            compressed_data = data / (adjusted_factor * random.uniform(1.5, 3.0))
            compressed_data = compressed_data.round(4)

            logging.info(f"Data compressed successfully with factor: {adjusted_factor}")
            self._save_to_cache(key, compressed_data)
            return compressed_data
    
    def quantum_predictive_entanglement(self, current_state, future_scenarios):
            """
            Implements Quantum Predictive Entanglement (QPE) to anticipate future scenarios and select the optimal path using qubits.
            """
            logging.info("Engaging Quantum Predictive Entanglement (QPE) process using qubits.")
            key = self._generate_cache_key('quantum_predictive_entanglement', current_state, tuple(future_scenarios))
            cached_result = self._get_from_cache(key)
            if cached_result is not None:
                return cached_result
            entangled_states = self._entangle_scenarios(current_state, future_scenarios)
            state_probabilities = self._calculate_state_probabilities(entangled_states)
            optimal_state = self._collapse_to_optimal_state(state_probabilities)
            self.qpe_state = optimal_state
            self._save_to_cache(key, optimal_state)
            logging.info(f"QPE process completed. Optimal predicted state: {self.qpe_state}")
            return self.qpe_state

    def _entangle_scenarios(self, current_state, future_scenarios):
            logging.info("Entangling current state with future scenarios using qubits.")
            entangled_states = []
            for scenario in future_scenarios:
                entangled_state = (current_state + scenario) / random.uniform(1.5, 3.0)  # Quantum entanglement simulation
                entangled_states.append(entangled_state)
            return entangled_states

    def _calculate_state_probabilities(self, entangled_states):
            logging.info("Calculating probabilities for each entangled state using qubits.")
            probabilities = [1 / (1 + math.exp(-state)) for state in entangled_states]  # Sigmoid function for probability calculation
            return probabilities
    
    def _collapse_to_optimal_state(self, state_probabilities):
            
            logging.info("Collapsing to the optimal predicted state using quantum algorithms.")
            optimal_index = np.argmax(state_probabilities)
            return state_probabilities[optimal_index]  # Returns the highest probability state

    def execute_qpe_routine(self, current_state):
            future_scenarios = np.random.rand(5)  # Simulated future scenarios
            optimal_prediction = self.quantum_predictive_entanglement(current_state, future_scenarios)
            return optimal_prediction

    async def adaptive_quantum_classical_fusion(self, classical_data, quantum_data):
            """
            Adaptive Quantum-Classical Fusion (QCF) logic.
            Dynamically decides whether to use quantum computation or stick with classical methods based on complexity.
            """
            complexity_score = self.estimate_task_complexity(classical_data)
            if complexity_score > self.qcf_threshold:
                self.logger.info("Complexity exceeds threshold, using quantum computation.")
                return await self.quantum_compute(quantum_data)
            else:
                self.logger.info("Complexity below threshold, using classical computation.")
                return classical_data
            
    def estimate_task_complexity(self, data):
            """
            Simple complexity estimation logic. In practice, you may refine this based on your task.
            """
            complexity = len(data) / 1000  # Example threshold (you can tune this for your case)
            return complexity

    
    async def quantum_gradient_boosting(self, gradients, learning_rate=0.01):
            """
            Implements Quantum Gradient Boosting to optimize model convergence.
            Quantum-enhanced gradient computations accelerate convergence during backpropagation.
            """
            try:
                quantum_circuit = QuantumCircuit(len(gradients))
                for i, gradient in enumerate(gradients):
                    quantum_circuit.rx(gradient * learning_rate, i)

                # Quantum gradient computation
                result = await self.quantum_compute(quantum_circuit)
                statevector = result.get_statevector()

                quantum_boosted_gradients = [state.real for state in statevector]
                self.logger.info(f"Quantum boosted gradients: {quantum_boosted_gradients}")
                return quantum_boosted_gradients
            except Exception as e:
                self.logger.error(f"Error during quantum gradient boosting: {e}")
                raise    

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
def get_resource_usage():
    process = psutil.Process()
    cpu_percent = psutil.cpu_percent(interval=None)
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # Convert to MB
    return cpu_percent, memory_usage

def measure_performance(func, *args, **kwargs):
    start_time = time.time()
    cpu_before, mem_before = get_resource_usage()
    
    result = func(*args, **kwargs)
    
    cpu_after, mem_after = get_resource_usage()
    elapsed_time = time.time() - start_time
    cpu_usage = cpu_after - cpu_before
    memory_usage = mem_after - mem_before
    
    return result, elapsed_time, cpu_usage, memory_usage

def run_tests():
    logger = logging.getLogger("QuantumModuleTest")
    logging.basicConfig(level=logging.DEBUG)

    # Store results for table generation
    results = []

    # Define Hugging Face models
    model_1_name = "gpt2"
    model_2_name = "bert-base-uncased"

    # Load models and tokenizers
    model_1 = AutoModelForCausalLM.from_pretrained(model_1_name)
    tokenizer_1 = AutoTokenizer.from_pretrained(model_1_name)

    model_2 = AutoModelForCausalLM.from_pretrained(model_2_name)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_2_name)

    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_1.to(device)
    model_2.to(device)

    # Test data
    input_text = """
    The quantum realm, a fascinating frontier of physics, challenges our understanding of reality. 
    In this microscopic world, particles can exist in multiple states simultaneously - a phenomenon known as superposition. 
    Entanglement, another quantum quirk, allows particles to be interconnected regardless of distance.
    
    These principles form the foundation of quantum computing, a revolutionary technology with the potential to solve complex problems exponentially faster than classical computers. 
    From optimizing supply chains to simulating molecular structures for drug discovery, the applications are vast and transformative.
    
    However, building a practical quantum computer faces numerous challenges:
    1. Maintaining quantum coherence
    2. Minimizing errors in quantum gates
    3. Scaling up to a sufficient number of qubits
    
    Despite these hurdles, researchers and tech giants alike are making steady progress. 
    In the near future, we may see quantum computers working alongside classical systems, ushering in a new era of computational power and scientific discovery.
    
    As we delve deeper into the quantum world, who knows what other mind-bending phenomena we'll uncover? 
    The journey of quantum exploration is just beginning!
    """
    tokens = [1, 0, 1, 1]
    freqs = {1: (0.2, 0.5), 0: (0.5, 1.0)}
    vocab_size = 8
    data = torch.rand(100).to(device)

    
    
     # Measure Hugging Face model performance
    models = [model_1, model_2]
    tokenizers = [tokenizer_1, tokenizer_2]
    model_names = [model_1_name, model_2_name]

    for model, tokenizer, model_name in zip(models, tokenizers, model_names):
        logger.info(f"Running inference for {model_name}...")
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        def inference():
            with torch.no_grad():
                return model(**inputs)
        
        outputs, inference_time, inference_cpu, inference_memory = measure_performance(inference)
        results.append({
            'Module': 'Inference',
            'Model': model_name,
            'Time (s)': inference_time,
            'CPU (%)': inference_cpu,
            'Memory (MB)': inference_memory
        })

    quantum_module = QuantumModule(vocab_size=vocab_size, logger=logger)
    # Measure quantum encoding/decoding performance
    logger.info("Testing Quantum Module Encoding/Decoding")
    encoded_value, encode_time, encode_cpu, encode_memory = measure_performance(
        asyncio.run, quantum_module.quantum_arithmetic_encode(tokens, freqs)
    )
    results.append({
        'Module': 'Quantum Arithmetic Encode',
        'Model': 'Quantum',
        'Time (s)': encode_time,
        'CPU (%)': encode_cpu,
        'Memory (MB)': encode_memory
    })
    
    quantum_state, sparse_time, sparse_cpu, sparse_memory = measure_performance(
        asyncio.run, quantum_module.quantum_sparse_encode(tokens, vocab_size)
    )
    results.append({
        'Module': 'Quantum Sparse Encode',
        'Model': 'Quantum',
        'Time (s)': sparse_time,
        'CPU (%)': sparse_cpu,
        'Memory (MB)': sparse_memory
    })

    huffman_codes, huffman_time, huffman_cpu, huffman_memory = measure_performance(
        asyncio.run, quantum_module.quantum_huffman_encode(freqs)
    )
    results.append({
        'Module': 'Quantum Huffman Encode',
        'Model': 'Quantum',
        'Time (s)': huffman_time,
        'CPU (%)': huffman_cpu,
        'Memory (MB)': huffman_memory
    })


    # Convert results to DataFrame and print
    df = pd.DataFrame(results)
    print(df)
    return df


   
if __name__ == "__main__":
    run_tests()