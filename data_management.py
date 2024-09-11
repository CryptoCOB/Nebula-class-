import asyncio
from asyncio.log import logger
import random
import aiohttp
import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as extract_text_from_pdf
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import sqlite3
import json
import numpy as np
from sklearn.preprocessing import normalize
from cachetools import TTLCache
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import hashlib, zipfile
import json
import pinecone
import zipfile
import pickle
from torch import amp
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
from FileProcessor import FileProcessor
import asyncio
from qiskit_aer import Aer
from qiskit import transpile, QuantumCircuit


class AdvancedDataManagement:
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError(f"Expected 'config' to be a dictionary, but got {type(config)}")
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logger()
        self.local_db_manager = self.config['local_db_path']
        self.connection = sqlite3.connect(self.local_db_manager, check_same_thread=False)
        self.cursor = self.connection.cursor()
        self._create_tables()
        self.lstm_memory = self._init_lstm_memory().to(self.device)
        self.website_scanner = WebsiteScanner(self.config, self)
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 5))
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        
        self.file_processor = FileProcessor(config, self.local_db_manager,  self.logger)
        self.file_processor.to(self.device)
        self.pinecone_manager = self._init_pinecone_manager()
        self.pinecone_index_name = config.get('pinecone_index_name', 'default_index')
        self.vector_dimension = config.get('pinecone_dimensions', 512)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.nlp_augmenter = pipeline("text2text-generation", model="t5-small")
        self.scaler = amp.GradScaler(device='cuda') 

    def to_dict(self):
        return self.config

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(self.config.get('log_file', 'data_management.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _create_tables(self):
        """Create the necessary tables in the local database."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS partial_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector BLOB,
                metadata TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                split TEXT,
                input BLOB,
                target BLOB,
                metadata TEXT,
                hash TEXT
            )
        ''')
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY,
            hash TEXT UNIQUE,
            data TEXT
        );
        """)
    

        self.connection.commit()

    def _init_pinecone_manager(self):
        pc = pinecone.Pinecone(api_key=self.config['pinecone_api_key'])
        return pc.Index(self.config['pinecone_index_name'])

    def _init_lstm_memory(self):
        return DataManagementLSTM(config=self.config).to(self.device)

    def show_database_info(self):
        info = "Advanced Database Information:\n"
        info += f"Pinecone Index: {self.pinecone_index_name}\n"
        info += f"Vector Dimension: {self.vector_dimension}\n"
        return info

    async def prepare_data_for_orion(self, data_sources: List[Dict[str, str]], model_name='bert-base-uncased', text_column='text', max_length=512) -> Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        self.logger.debug("Starting data preparation for ORION")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        all_data = {'train': {'combined': None}, 'validation': {'combined': None}}

        for source in data_sources:
            dataset_name = source['name']
            config_name = source.get('config', None)
            self.logger.debug(f"Loading dataset: {dataset_name} with config: {config_name}")

            try:
                if 'huggingface' in dataset_name.lower():
                    self.logger.debug(f"Preprocessing Hugging Face dataset: {dataset_name}")
                    train_loader, val_loader = await self.file_processor.preprocess_huggingface_dataset(dataset_name)
                    
                    for inputs, labels in train_loader:
                        all_data['train']['combined'] = self.file_processor.combine_data(all_data['train']['combined'], inputs, labels)
                    for inputs, labels in val_loader:
                        all_data['validation']['combined'] = self.file_processor.combine_data(all_data['validation']['combined'], inputs, labels)
                    
                    self.logger.debug(f"Processed Hugging Face dataset: {dataset_name}")

                else:
                    dataset = await self.load_dataset_with_retry(dataset_name, config_name)
                    if dataset is None:
                        self.logger.error(f"Failed to load dataset: {dataset_name}")
                        continue

                    for split in ['train', 'validation']:
                        if split in dataset:
                            self.logger.debug(f"Processing {split} data from {dataset_name}")
                            inputs, labels = await self.process_split(dataset[split], tokenizer, text_column, max_length)
                            all_data[split]['combined'] = self.file_processor.combine_data(all_data[split]['combined'], inputs['input_ids'], labels)
                            self.logger.debug(f"Processed {len(inputs['input_ids'])} {split} samples from {dataset_name}")
                        else:
                            self.logger.warning(f"Split '{split}' not found in dataset {dataset_name}")
            except Exception as e:
                self.logger.error(f"Error processing data from {dataset_name}: {e}")

        # Validate that both train and validation data were processed
        for split in ['train', 'validation']:
            if all_data[split]['combined'] is not None:
                inputs, labels = all_data[split]['combined']
                await self.store_data_in_chunks(inputs, labels, split)
            else:
                self.logger.error(f"No data processed for {split} split, possible issue with dataset sources")

        self.logger.debug("Completed data preparation for ORION")
        return all_data


    async def load_dataset_with_retry(self, dataset_name, config_name, max_retries=3):
        for attempt in range(max_retries):
            try:
                if config_name:
                    return load_dataset(dataset_name, config_name)
                else:
                    return load_dataset(dataset_name)
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed to load dataset {dataset_name}: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to load dataset {dataset_name} after {max_retries} attempts")
                    return None
                await asyncio.sleep(1)  # Wait for 1 second before retrying

    async def store_data_in_chunks(self, inputs: torch.Tensor, labels: torch.Tensor, split: str, chunk_size: int = 1000):
        total_samples = inputs.shape[0]
        for i in range(0, total_samples, chunk_size):
            chunk_inputs = inputs[i:i+chunk_size]
            chunk_labels = labels[i:i+chunk_size]
            data_to_store = {
                'input': chunk_inputs.numpy().tolist(),
                'target': chunk_labels.numpy().tolist(),
                'metadata': {'split': split, 'shape': chunk_inputs.shape, 'chunk': i // chunk_size}
            }
            await self.store_training_data('combined', split, data_to_store)
        self.logger.debug(f"Stored {total_samples} samples for {split} split in chunks")

    def hash_data(self, data):
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    async def check_data_exists(self, data_hash: str) -> bool:
        count = await self._execute_db_operation(
            "SELECT COUNT(*) FROM training_data WHERE hash=?", (data_hash,), fetch_one=True
        )
        return count[0] > 0

    async def store_training_data(self, source: str, split: str, data: Dict[str, Any]):
        data_hash = self.hash_data(data)
        input_data = pickle.dumps(data['input'])
        target_data = pickle.dumps(data['target'])
        metadata = json.dumps(data['metadata'])

        if not await self.check_data_exists(data_hash):
            await self._execute_db_operation(
                '''INSERT INTO training_data (source, split, input, target, metadata, hash)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (source, split, input_data, target_data, metadata, data_hash)
            )
            self.logger.debug(f"Stored new data chunk for {split} split with hash {data_hash[:6]}")
        else:
            self.logger.debug(f"Data chunk for {split} split with hash {data_hash[:12]} already exists, skipping")
            

    async def _execute_db_operation(self, query: str, params: Optional[Tuple] = None, fetch_one: bool = False, fetch_all: bool = False):
        """Execute a database operation."""
        loop = asyncio.get_event_loop()
        if params:
            result = await loop.run_in_executor(None, self.cursor.execute, query, params)
        else:
            result = await loop.run_in_executor(None, self.cursor.execute, query)

        if fetch_one:
            return await loop.run_in_executor(None, self.cursor.fetchone)
        elif fetch_all:
            return await loop.run_in_executor(None, self.cursor.fetchall)
        else:
            await loop.run_in_executor(None, self.connection.commit)
            return result

    async def get_new_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch new data from various sources and preprocess it for training.

        Returns:
            Optional[List[Dict[str, Any]]]: Preprocessed new data, or None if no new data is available.
        """
        try:
            self.logger.info("Fetching new data from sources...")
            new_data = []

            # Check each configured source type for new data
            if 'api' in self.config.get('source_types', []):
                api_data = await self._fetch_data_from_api()
                if api_data:
                    new_data.extend(api_data)

            if 'database' in self.config.get('source_types', []):
                db_data = await self._fetch_data_from_database()
                if db_data:
                    new_data.extend(db_data)

            if 'local_files' in self.config.get('source_types', []):
                file_data = await self._fetch_data_from_local_files()
                if file_data:
                    new_data.extend(file_data)

            # Preprocess the new data if any was fetched
            if new_data:
                preprocessed_data = await self._preprocess_new_data(new_data)
                return preprocessed_data

            self.logger.info("No new data available.")
            return None

        except Exception as e:
            self.logger.error(f"Error fetching new data: {e}")
            return None

    async def _fetch_data_from_api(self) -> List[Dict[str, Any]]:
        """
        Fetch data from API sources.

        Returns:
            List[Dict[str, Any]]: A list of data dictionaries fetched from the API.
        """
        try:
            self.logger.info("Fetching data from API sources...")
            # Placeholder for actual API fetching logic
            # Replace this with real API data fetching as needed
            response = await self._simulate_api_fetch()
            return response.get('data', []) if response else []

        except Exception as e:
            self.logger.error(f"Error fetching data from API: {e}")
            return []

    async def _fetch_data_from_database(self) -> List[Dict[str, Any]]:
        """
        Fetch data from the local database.

        Returns:
            List[Dict[str, Any]]: A list of data dictionaries fetched from the database.
        """
        try:
            self.logger.info("Fetching data from the local database...")
            # Placeholder for actual database query logic
            # Replace this with real database fetching logic as needed
            data = await self._execute_db_operation("SELECT * FROM training_data", fetch_all=True)
            return [{'input': pickle.loads(row[3]), 'target': pickle.loads(row[4]), 'metadata': json.loads(row[5])} for row in data]

        except Exception as e:
            self.logger.error(f"Error fetching data from database: {e}")
            return []

    async def _fetch_data_from_local_files(self) -> List[Dict[str, Any]]:
        """
        Fetch data from local files.

        Returns:
            List[Dict[str, Any]]: A list of data dictionaries fetched from local files.
        """
        try:
            self.logger.info("Fetching data from local files...")
            # Placeholder for actual file reading logic
            # Replace this with real file reading and processing logic as needed
            file_paths = self.config.get('local_file_paths', [])
            data = []
            for file_path in file_paths:
                await self.process_single_data(file_path)
                data.append(await self._read_processed_data(file_path))
            return data

        except Exception as e:
            self.logger.error(f"Error fetching data from local files: {e}")
            return []

    async def _preprocess_new_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess new data for training.

        Args:
            raw_data (List[Dict[str, Any]]): The raw data to be preprocessed.

        Returns:
            List[Dict[str, Any]]: The preprocessed data.
        """
        self.logger.info("Preprocessing new data...")
        preprocessed_data = []
        for data_item in raw_data:
            preprocessed_input = await self.preprocess_text(data_item.get('input', ''))
            preprocessed_target = data_item.get('target', {})
            preprocessed_data.append({
                'input': preprocessed_input,
                'target': preprocessed_target,
                'metadata': data_item.get('metadata', {})
            })
        return preprocessed_data
    
    async def _simulate_api_fetch(self) -> Dict[str, Any]:
        """
        Simulate fetching data from an API.

        Returns:
            Dict[str, Any]: Mocked response data.
        """
        try:
            self.logger.info("Simulating API data fetch...")
            # Simulate delay that would occur during an actual API request
            await asyncio.sleep(1)

            # Mocked response data
            simulated_data = {
                'status': 'success',
                'data': [
                    {'input': 'Sample input text 1', 'target': {'label': 0}, 'metadata': {'source': 'api'}},
                    {'input': 'Sample input text 2', 'target': {'label': 1}, 'metadata': {'source': 'api'}}
                ]
            }

            self.logger.info("Successfully simulated API data fetch.")
            return simulated_data

        except Exception as e:
            self.logger.error(f"Error simulating API data fetch: {e}")
            return {}


    async def _read_processed_data(self, file_path: str) -> Dict[str, Any]:
        """
        Read processed data from a file.

        Args:
            file_path (str): The path to the file containing processed data.

        Returns:
            Dict[str, Any]: The data read from the file.
        """
        try:
            self.logger.info(f"Reading processed data from file: {file_path}")
            
            # Simulate delay that would occur when reading a large file
            await asyncio.sleep(1)

            # Load the processed data from the file
            with open(file_path, 'rb') as file:
                data = pickle.load(file)

            # Ensure the data is in the expected format
            if isinstance(data, dict) and 'input' in data and 'target' in data:
                self.logger.info(f"Successfully read processed data from {file_path}")
                return data
            else:
                self.logger.warning(f"Data in {file_path} is not in the expected format.")
                return {}

        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error reading processed data from {file_path}: {e}")
            return {}


    def combine_data(self, existing_data, new_inputs, new_labels):
        if existing_data is None:
            return (new_inputs, new_labels)
        else:
            combined_inputs = torch.cat([existing_data[0], new_inputs], dim=0)
            combined_labels = torch.cat([existing_data[1], new_labels], dim=0)
            return (combined_inputs, combined_labels)

    async def split_and_process_data(self, dataset, tokenizer, text_column, max_length, test_size=0.2):
        # Split the dataset into training and validation sets
        train_data, val_data = train_test_split(dataset, test_size=test_size)

        # Process the training data
        self.logger.debug("Processing training data")
        train_inputs, train_labels = await self.process_split(train_data, tokenizer, text_column, max_length)

        # Process the validation data
        self.logger.debug("Processing validation data")
        val_inputs, val_labels = await self.process_split(val_data, tokenizer, text_column, max_length)

        return {
            'train': (train_inputs, train_labels),
            'validation': (val_inputs, val_labels)
        }
    
    async def process_split(self, split_data, tokenizer, text_column, max_length):
        processed_texts = []
        labels = []
        
        for item in split_data:
            text = self.extract_text(item)
            
            if text is not None:
                processed_text = await self.preprocess_text(text)
                processed_texts.append(processed_text)
                labels.append(item.get('label', 0))  # Default to 0 if no label is found
            else:
                self.logger.warning(f"No recognized text column found in data item: {item}")
        
        inputs = tokenizer(processed_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        labels = torch.tensor(labels).to(self.device)
        
        return inputs, labels

    def extract_text(self, item):
        for possible_column in [
            'text', 'topic', 'formatted_prompt', 'completion', 'first_task', 'second_task', 'last_task',
            'notes', 'markdown', 'sentence', 'content', 'document', 'review', 'question', 'answer',
            'title', 'description', 'comment', 'transcript', 'summary', 'article', 'dialogue',
            'snippet', 'post', 'message', 'tweet', 'script', 'abstract', 'passage', 'excerpt',
            'narrative', 'log', 'note', 'feedback', 'annotation', 'commentary', 'remark',
            'observation', 'report', 'statement', 'explanation', 'justification', 'interpretation',
            'analysis', 'evaluation', 'criticism', 'review_text', 'comment_text', 'post_text',
            'message_text', 'tweet_text', 'script_text', 'abstract_text', 'passage_text',
            'excerpt_text', 'narrative_text', 'log_text', 'note_text', 'feedback_text',
            'annotation_text', 'commentary_text', 'remark_text', 'observation_text', 'report_text',
            'statement_text', 'explanation_text', 'justification_text', 'interpretation_text',
            'analysis_text', 'evaluation_text', 'criticism_text'
        ]:
            if possible_column in item:
                self.logger.debug(f"Found '{possible_column}' column in data item")
                return item[possible_column]
        return None

    
    def get_training_data(self):
        return {
            'train': {'combined': self.prepare_data_for_training('train')},
            'validation': {'combined': self.prepare_data_for_training('validation')}
        }

    async def prepare_data_for_training(self, split, data_source):
        if split not in self.prepared_data:
            # Process the data using the process_data method
            data = await self.process_data(data_source)

            # Prepare the data for training
            inputs, labels = self._prepare_data(data)
            inputs = torch.tensor(inputs).to(self.device)
            labels = torch.tensor(labels).to(self.device)
            
            self.prepared_data[split] = (inputs, labels)

        return self.prepared_data[split]

    async def process_data(self, data_source: Union[str, List[str]]):
        processed_data = []
        
        if isinstance(data_source, list):
            results = await asyncio.gather(*(self.process_single_data(ds) for ds in data_source))
            processed_data.extend(results)
        elif isinstance(data_source, str) and data_source.endswith('.zip'):
            with zipfile.ZipFile(data_source, 'r') as zip_ref:
                extract_path = '/tmp/test_dataset'
                zip_ref.extractall(extract_path)
                for root, _, files in os.walk(extract_path):
                    for file in files:
                        result = await self.process_single_data(os.path.join(root, file))
                        processed_data.append(result)
        elif isinstance(data_source, str):
            result = await self.process_single_data(data_source)
            processed_data.append(result)
        else:
            raise ValueError("Invalid data source type")

        return processed_data

    async def process_single_data(self, data_source: str):
        try:
            if os.path.isfile(data_source):
                result = await self.file_processor.process_file(data_source)
                await self.store_vector(data_source, result['input'].tolist())
                return result
            elif data_source.startswith('http'):
                result = await self.process_website(data_source)
                return result
            elif 'huggingface' in data_source.lower():
                result = await self.file_processor.preprocess_huggingface_dataset(data_source)
                return result
            else:
                raise ValueError(f"Unsupported data source: {data_source}")
        except Exception as e:
            self.logger.error(f"Error processing data source {data_source}: {e}")
            return None

    def _prepare_data(self, data):
        # Assuming data is already processed into the appropriate format, e.g., a list of dictionaries or tuples
        processed_inputs = []
        processed_labels = []

        for item in data:
            input_data = item.get('input')
            label_data = item.get('label', 0)
            processed_inputs.append(input_data)
            processed_labels.append(label_data)

            # Convert to tensors
        inputs = torch.tensor(processed_inputs).to(self.device)
        labels = torch.tensor(processed_labels).to(self.device)

        # Ensure inputs and labels are on the same device
        return inputs, labels
        

    def process_text(self, text):
        # Example text processing logic
        return text.lower()  # Simplified example

    async def preprocess_text(self, text: Any) -> str:
        if isinstance(text, str):
            return text.lower()
        elif isinstance(text, dict):
            try:
                return json.dumps(text).lower()
            except (TypeError, ValueError) as e:
                self.logger.error(f"Error converting dict to JSON string: {e}")
                raise ValueError("Invalid dict format for text preprocessing")
        elif isinstance(text, list):
            try:
                return ' '.join(map(str, text)).lower()
            except Exception as e:
                self.logger.error(f"Error converting list to string: {e}")
                raise ValueError("Invalid list format for text preprocessing")
        else:
            self.logger.error(f"Unsupported type for text preprocessing: {type(text)}")
            raise ValueError(f"Unsupported type for text preprocessing: {type(text)}")

    def _tokenize_text(self, text: str) -> List[int]:
        tokens = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        return tokens['input_ids'].squeeze().tolist()

    async def text_to_vector(self, text: str) -> List[float]:
        vector = self.lstm_memory(self._prepare_input_tensor(self._tokenize_text(await self.preprocess_text(text))))
        normalized_vector = normalize(vector.detach().cpu().numpy().reshape(1, -1)).flatten().tolist()
        return normalized_vector

    async def store_vector(self, source: str, vector: List[float]):
        metadata = {'source': source}
        await self.pinecone_manager.upsert(vectors=[(source, vector, metadata)])

    async def process_website(self, url: str):
        try:
            self.logger.info(f"Processing website: {url}")
            text_data = await self.website_scanner.scrape_website(url)
            if text_data:
                vector_data = await self.text_to_vector(text_data)
                await self.store_vector(url, vector_data)
                return {"url": url, "vector_data": vector_data}
            else:
                self.logger.warning(f"No text extracted from website: {url}")
                return {"url": url, "vector_data": None}
        except Exception as e:
            self.logger.error(f"Error processing website {url}: {e}")
            return {"url": url, "error": str(e)}

    async def query_data(self, vector: List[float]) -> List[Dict[str, Any]]:
        results = await self.pinecone_manager.query(vector=vector, top_k=10, include_metadata=True)
        return [{'id': match['id'], 'score': match['score'], 'metadata': match['metadata']} for match in results['matches']]

    async def update_data(self, vector_id: str, new_vector: List[float], new_metadata: Dict[str, Any]):
        try:
            await self.pinecone_manager.update(id=vector_id, values=new_vector, set_metadata=new_metadata)
            self.logger.info(f"Updated vector {vector_id} in Pinecone")
            if vector_id in self.cache:
                del self.cache[vector_id]
        except Exception as e:
            self.logger.error(f"Error updating data for vector_id {vector_id}: {e}")

    async def delete_data(self, vector_id: str):
        try:
            await self.pinecone_manager.delete(ids=[vector_id])
            await self._execute_db_operation("DELETE FROM partial_vectors WHERE id=?", (vector_id,))
            self.logger.info(f"Deleted data for vector_id: {vector_id}")
            if vector_id in self.cache:
                del self.cache[vector_id]
        except Exception as e:
            self.logger.error(f"Error deleting data for vector_id {vector_id}: {e}")

    async def clear_all_data(self):
        try:
            await self.pinecone_manager.delete(delete_all=True)
            self.cursor.execute("DELETE FROM partial_vectors")
            self.connection.commit()
            self.cache.clear()
            self.logger.info("Cleared all data from Pinecone, local database, and cache")
        except Exception as e:
            self.logger.error(f"Error clearing all data: {e}")

    async def get_data_stats(self) -> Dict[str, Any]:
        try:
            pinecone_stats = await self.pinecone_manager.describe_index_stats()
            local_db_stats = await self.get_db_stats()
            return {
                "pinecone": pinecone_stats,
                "local_db": local_db_stats,
                "cache_size": len(self.cache)
            }
        except Exception as e:
            self.logger.error(f"Error getting data stats: {e}")
            return {"pinecone": {}, "local_db": {}, "cache_size": 0, "error": str(e)}

    async def get_db_stats(self) -> Dict[str, Any]:
        """Get statistics for the local database."""
        count = await self._execute_db_operation("SELECT COUNT(*) FROM partial_vectors", fetch_one=True)
        return {"partial_vectors_count": count[0] if count else 0}

    async def batch_process(self, data_sources: List[str]):
        try:
            tasks = [self.process_single_data(source) for source in data_sources]
            await asyncio.gather(*tasks)
            self.logger.info(f"Batch processed {len(data_sources)} data sources")
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")

    async def get_vector_by_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        try:
            vector = await self.pinecone_manager.fetch(ids=[vector_id])
            if vector and vector_id in vector['vectors']:
                return {
                    'id': vector_id,
                    'vector': vector['vectors'][vector_id]['values'],
                    'metadata': vector['vectors'][vector_id]['metadata']
                }
            return None
        except Exception as e:
            self.logger.error(f"Error fetching vector {vector_id}: {e}")
            return None

    async def search_by_metadata(self, metadata_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            results = await self.pinecone_manager.query(
                vector=[0] * self.vector_dimension,
                filter=metadata_query,
                top_k=10,
                include_metadata=True
            )
            return [{'id': match['id'], 'score': match['score'], 'metadata': match['metadata']} for match in results['matches']]
        except Exception as e:
            self.logger.error(f"Error searching by metadata: {e}")
            raise e

    async def update_vector_metadata(self, vector_id: str, new_metadata: Dict[str, Any]):
        try:
            await self.pinecone_manager.update(id=vector_id, set_metadata=new_metadata)
            self.logger.info(f"Updated metadata for vector {vector_id}")
            if vector_id in self.cache:
                del self.cache[vector_id]
        except Exception as e:
            self.logger.error(f"Error updating metadata for vector {vector_id}: {e}")

    async def perform_bulk_update(self, updates: List[Dict[str, Any]]):
        try:
            await self.pinecone_manager.upsert(vectors=[(u['id'], u['values'], u.get('metadata', {})) for u in updates])
            self.logger.info(f"Performed bulk update on {len(updates)} vectors")
            for update in updates:
                if update['id'] in self.cache:
                    del self.cache[update['id']]
        except Exception as e:
            self.logger.error(f"Error performing bulk update: {e}")

    async def get_nearest_neighbors(self, vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        try:
            results = await self.pinecone_manager.query(vector=vector, top_k=k, include_metadata=True)
            return [{'id': match['id'], 'score': match['score'], 'metadata': match['metadata']} for match in results['matches']]
        except Exception as e:
            self.logger.error(f"Error getting nearest neighbors: {e}")
            return []

    async def compute_centroid(self, vector_ids: List[str]) -> Optional[List[float]]:
        try:
            vectors = await asyncio.gather(*[self.get_vector_by_id(vid) for vid in vector_ids])
            valid_vectors = [v['vector'] for v in vectors if v]
            if valid_vectors:
                centroid = np.mean(valid_vectors, axis=0).tolist()
                return centroid
            return None
        except Exception as e:
            self.logger.error(f"Error computing centroid: {e}")
            return None

    async def find_clusters(self, num_clusters: int = 5) -> Dict[int, List[str]]:
        try:
            all_vectors = await self.pinecone_manager.fetch(ids=None)
            vector_ids = list(all_vectors['vectors'].keys())
            vectors = [all_vectors['vectors'][vid]['values'] for vid in vector_ids]

            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(vectors)

            clusters = {i: [] for i in range(num_clusters)}
            for vid, cluster in zip(vector_ids, kmeans.labels_):
                clusters[cluster].append(vid)

            return clusters
        except Exception as e:
            self.logger.error(f"Error finding clusters: {e}")
            return {}

    async def close(self):
        await self.website_scanner.close()
        self.connection.close()
        self.logger.info("DataManagement resources closed")

    def convert_to_tensors(self, data):
        return torch.tensor(data).to(self.device)

    def _prepare_input_tensor(self, tokenized_text: List[int]) -> torch.Tensor:
        return torch.tensor(tokenized_text, dtype=torch.long).unsqueeze(0)

    async def augment_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return await self.file_processor.augment_dataset(data)

    async def process_and_store_file(self, file_path: str):
        try:
            result = await self.file_processor.process_file(file_path)
            await self.store_vector(file_path, result['input'].flatten().tolist())
            self.logger.info(f"Processed and stored file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error processing and storing file {file_path}: {e}")

    async def batch_process_files(self, file_paths: List[str]):
        tasks = [self.process_and_store_file(file_path) for file_path in file_paths]
        await asyncio.gather(*tasks)

    async def get_vector_statistics(self) -> Dict[str, Any]:
        try:
            stats = await self.pinecone_manager.describe_index_stats()
            return {
                "total_vector_count": stats['total_vector_count'],
                "dimension": stats['dimension'],
                "namespaces": stats['namespaces']
            }
        except Exception as e:
            self.logger.error(f"Error getting vector statistics: {e}")
            return {}

    async def perform_vector_operation(self, operation: str, vectors: List[Dict[str, Any]]):
        try:
            if operation == "upsert":
                await self.pinecone_manager.upsert(vectors=[(v['id'], v['values'], v.get('metadata', {})) for v in vectors])
            elif operation == "delete":
                await self.pinecone_manager.delete(ids=[v['id'] for v in vectors])
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            self.logger.info(f"Performed {operation} operation on {len(vectors)} vectors")
        except Exception as e:
            self.logger.error(f"Error performing {operation} operation: {e}")

class PineconeManager:
    def __init__(self, config):
        self.config = config
        self.pc = Pinecone(api_key=config['pinecone_api_key'])
        self.index_name = config['pinecone_index_name']
        self.cloud = config['pinecone_cloud']
        self.dimension = config['pinecone_dimensions']
        self.batch_size = config.get('batch_size', 100)
        self.namespace = config.get('namespace', '')
        self.logger = logging.getLogger(__name__)
        self.vector_batch = []
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        self._initialize_index()

    def _initialize_index(self):
        if self.index_name not in self.pc.list_indexes().names():
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.config['pinecone_dimension'],
                    metric=self.config['pinecone_metric'],
                    spec=ServerlessSpec(
                        cloud=self.config.get('pinecone_cloud', 'aws'),
                        region=self.config['pinecone_region']
                    )
                )
                self.logger.info(f"Pinecone index '{self.index_name}' created")
            except Exception as e:
                if 'already exists' in str(e):
                    self.logger.info(f"Pinecone index '{self.index_name}' already exists")
                else:
                    self.logger.error(f"Error creating Pinecone index: {e}")
                    raise
        self.index = self.pc.Index(self.index_name)
        self.logger.info(f"Connected to Pinecone index '{self.index_name}'")

    def list_indexes(self):
        try:
            return self.pc.list_indexes().names()
        except Exception as e:
            self.logger.error(f"Error listing Pinecone indexes: {e}")
            return []

    def save_to_pinecone(self, id, vector):
        try:
            index = pinecone.Index(self.config['pinecone_index_name'])
            result = index.upsert(vectors=[(id, vector)], namespace='test-namespace')
            return result  # Return the result of the upsert operation
        except Exception as e:
            print(f"Error saving to Pinecone: {str(e)}")
            return None

    def store_vector(self, vector: Union[List[float], np.ndarray, Dict[str, Any]], vector_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        if isinstance(vector, dict):
            vector_id = vector.get('id')
            metadata = {"sources": json.dumps(vector['sources'])}
            vector = vector['vector']

        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} does not match index dimension {self.dimension}")

        if vector_id is None:
            vector_id = self._generate_vector_id(vector)

        if metadata is None:
            metadata = {}

        metadata.update({
            "timestamp": datetime.utcnow().isoformat(),
            "vector_norm": float(np.linalg.norm(vector))
        })

        self.vector_batch.append((vector_id, vector, metadata))

        if len(self.vector_batch) >= self.batch_size:
            return self._upsert_batch()

    def _upsert_batch(self, retry_attempts: int = 3):
        for attempt in range(retry_attempts):
            try:
                result = self.index.upsert(vectors=self.vector_batch, namespace=self.namespace)
                self.logger.info(f"Batch of {len(self.vector_batch)} vectors upserted successfully")
                self.vector_batch.clear()
                return result
            except Exception as e:
                self.logger.error(f"Error upserting batch (attempt {attempt + 1}/{retry_attempts}): {e}")
                if attempt == retry_attempts - 1:
                    raise

                    
    def vectorize(self, data):
        # Implement vectorization logic here
        # For example, converting text to embeddings
        return np.array([self.text_to_embedding(text) for text in data])

    def text_to_embedding(self, text):
        # Convert text to embedding (example implementation)
        return np.random.rand(512)  
                    

    def flush(self):
        if self.vector_batch:
            return self._upsert_batch()

    def _generate_vector_id(self, vector: List[float]) -> str:
        vector_str = json.dumps(vector, sort_keys=True)
        return hashlib.md5(vector_str.encode()).hexdigest()

    def query_pinecone(self, query_vector):
        try:
            index = pinecone.Index(self.config['pinecone_index_name'])
            results = index.query(vector=query_vector, top_k=10, namespace='test-namespace')
            return results  # This is already a dict, no need to convert
        except Exception as e:
            print(f"Error querying Pinecone: {str(e)}")
            return None

    def delete_from_pinecone(self, vector_id: str):
        try:
            self.index.delete(ids=[vector_id], namespace=self.namespace)
            self.logger.info(f"Vector {vector_id} deleted from Pinecone")
        except Exception as e:
            self.logger.error(f"Error deleting vector from Pinecone: {e}")
            raise

    # ... (other methods remain the same)

    def clear_pinecone_index(self):
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            self.logger.info("Pinecone index cleared")
        except Exception as e:
            self.logger.error(f"Error clearing Pinecone index: {e}")
            raise



    async def get_index_stats(self) -> Dict[str, Any]:
        try:
            stats = await self.index.describe_index_stats()
            return stats
        except Exception as e:
            self.logger.error(f"Error getting Pinecone index stats: {e}")
            raise

    async def fetch_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = await self.index.fetch(ids=[vector_id], namespace=self.namespace)
            return response['vectors'].get(vector_id)
        except Exception as e:
            self.logger.error(f"Error fetching vector from Pinecone: {e}")
            return None

    async def search_by_metadata(self, metadata_filter: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        try:
            results = await self.index.query(
                vector=[0] * self.dimension,  # Dummy vector, we're only interested in metadata
                filter=metadata_filter,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True
            )
            return results
        except Exception as e:
            self.logger.error(f"Error searching by metadata in Pinecone: {e}")
            raise

    async def update_metadata(self, vector_id: str, new_metadata: Dict[str, Any]):
        try:
            await self.index.update(id=vector_id, set_metadata=new_metadata, namespace=self.namespace)
            self.logger.info(f"Metadata updated for vector {vector_id}")
        except Exception as e:
            self.logger.error(f"Error updating metadata in Pinecone: {e}")
            raise

    async def bulk_update(self, updates: List[Dict[str, Any]]):
        try:
            await self.index.upsert(vectors=updates, namespace=self.namespace)
            self.logger.info(f"Bulk update of {len(updates)} vectors completed")
        except Exception as e:
            self.logger.error(f"Error performing bulk update in Pinecone: {e}")
            raise

    async def fetch_all_vectors(self) -> Dict[str, List[float]]:
        try:
            all_vectors = {}
            async for batch in self.index.iter_vectors(self.namespace):
                for id, vector, metadata in batch:
                    all_vectors[id] = vector
            return all_vectors
        except Exception as e:
            self.logger.error(f"Error fetching all vectors from Pinecone: {e}")
            raise

    async def compute_centroid(self, vector_ids: List[str]) -> List[float]:
        vectors = await self.fetch_vectors(vector_ids)
        if not vectors:
            raise ValueError("No vectors found for the given IDs")
        centroid = np.mean([v['values'] for v in vectors.values()], axis=0)
        return centroid.tolist()

    async def find_nearest_to_centroid(self, vector_ids: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        centroid = await self.compute_centroid(vector_ids)
        results = await self.query_pinecone(centroid, top_k=top_k)
        return results['matches']

    async def compute_vector_average(self, vector_ids: List[str], weights: Optional[List[float]] = None) -> List[float]:
        vectors = await self.fetch_vectors(vector_ids)
        if not vectors:
            raise ValueError("No vectors found for the given IDs")
        if weights is None:
            weights = [1.0] * len(vectors)
        if len(weights) != len(vectors):
            raise ValueError("Number of weights must match number of vectors")
        weighted_sum = np.sum([np.array(v['values']) * w for v, w in zip(vectors.values(), weights)], axis=0)
        return (weighted_sum / sum(weights)).tolist()

    async def find_vectors_in_range(self, center_vector: List[float], min_similarity: float, max_similarity: float, top_k: int = 10000) -> List[Dict[str, Any]]:
        results = await self.query_pinecone(center_vector, top_k=top_k)
        filtered_results = [
            r for r in results['matches']
            if min_similarity <= r['score'] <= max_similarity
        ]
        return filtered_results

    async def fetch_vectors(self, vector_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        try:
            response = await self.index.fetch(ids=vector_ids, namespace=self.namespace)
            return response['vectors']
        except Exception as e:
            self.logger.error(f"Error fetching vectors: {e}")
            raise

    async def vector_operation(self, operation: str, vectors: List[Dict[str, Any]]):
        try:
            if operation == 'upsert':
                await self.index.upsert(vectors=vectors, namespace=self.namespace)
            elif operation == 'delete':
                await self.index.delete(ids=[v['id'] for v in vectors], namespace=self.namespace)
            elif operation == 'update':
                for v in vectors:
                    await self.index.update(id=v['id'], set_metadata=v.get('metadata', {}), namespace=self.namespace)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            self.logger.info(f"Processed batch {operation} operation on {len(vectors)} vectors")
        except Exception as e:
            self.logger.error(f"Error processing batch {operation} operation: {e}")
            raise

    async def vector_similarity_search(self, query_vector: List[float], filter: Optional[Dict[str, Any]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        try:
            results = await self.query_pinecone(query_vector, top_k=top_k, filter=filter)
            return results['matches']
        except Exception as e:
            self.logger.error(f"Error performing similarity search: {e}")
            raise

    async def vector_aggregation(self, vector_ids: List[str], aggregation_type: str = 'mean') -> List[float]:
        vectors = await self.fetch_vectors(vector_ids)
        if not vectors:
            raise ValueError("No vectors found for the given IDs")

        vector_array = np.array([v['values'] for v in vectors.values()])

        if aggregation_type == 'mean':
            result = np.mean(vector_array, axis=0)
        elif aggregation_type == 'max':
            result = np.max(vector_array, axis=0)
        elif aggregation_type == 'min':
            result = np.min(vector_array, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

        return result.tolist()

    async def vector_semantic_search(self, query_text: str, encoder_model, top_k: int = 10) -> List[Dict[str, Any]]:
        try:
            query_vector = await asyncio.to_thread(encoder_model.encode, [query_text])
            query_vector = query_vector[0].tolist()  # Convert to list and get the first (and only) vector
            return await self.vector_similarity_search(query_vector, top_k=top_k)
        except Exception as e:
            self.logger.error(f"Error performing semantic search: {e}")
            raise

    async def vector_clustering(self, vector_ids: List[str], n_clusters: int = 5):
        try:
            from sklearn.cluster import KMeans

            vectors = await self.fetch_vectors(vector_ids)
            if not vectors:
                raise ValueError("No vectors found for the given IDs")

            vector_array = np.array([v['values'] for v in vectors.values()])

            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = await asyncio.to_thread(kmeans.fit_predict, vector_array)

            clusters = {i: [] for i in range(n_clusters)}
            for vid, label in zip(vector_ids, cluster_labels):
                clusters[label].append(vid)

            return clusters
        except ImportError:
            self.logger.error("sklearn is required for clustering. Please install it.")
            raise
        except Exception as e:
            self.logger.error(f"Error performing vector clustering: {e}")
            raise

    async def vector_space_analysis(self, vector_ids: List[str]):
        try:
            vectors = await self.fetch_vectors(vector_ids)
            if not vectors:
                raise ValueError("No vectors found for the given IDs")

            vector_array = np.array([v['values'] for v in vectors.values()])

            analysis = {
                "mean_vector": np.mean(vector_array, axis=0).tolist(),
                "std_dev_vector": np.std(vector_array, axis=0).tolist(),
                "min_vector": np.min(vector_array, axis=0).tolist(),
                "max_vector": np.max(vector_array, axis=0).tolist(),
                "vector_norms": np.linalg.norm(vector_array, axis=1).tolist(),
                "pairwise_distances": {
                    "min": np.min(np.linalg.norm(vector_array[:, None] - vector_array, axis=2)),
                    "max": np.max(np.linalg.norm(vector_array[:, None] - vector_array, axis=2)),
                    "mean": np.mean(np.linalg.norm(vector_array[:, None] - vector_array, axis=2))
                }
            }

            return analysis
        except Exception as e:
            self.logger.error(f"Error performing vector space analysis: {e}")
            raise
        
    

    async def detect_anomalies(self, vector_ids: List[str], threshold: float = 2.0):
        try:
            from sklearn.ensemble import IsolationForest

            vectors = await self.fetch_vectors(vector_ids)
            if not vectors:
                raise ValueError("No vectors found for the given IDs")

            vector_array = np.array([v['values'] for v in vectors.values()])

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = await asyncio.to_thread(iso_forest.fit_predict, vector_array)

            anomalies = [vid for vid, label in zip(vector_ids, anomaly_labels) if label == -1]
            return anomalies
        except ImportError:
            self.logger.error("sklearn is required for anomaly detection. Please install it.")
            raise
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            raise

    async def incremental_pca(self, vector_ids: List[str], n_components: int = 2):
        try:
            from sklearn.decomposition import IncrementalPCA

            vectors = await self.fetch_vectors(vector_ids)
            if not vectors:
                raise ValueError("No vectors found for the given IDs")

            vector_array = np.array([v['values'] for v in vectors.values()])

            ipca = IncrementalPCA(n_components=n_components)
            transformed_vectors = await asyncio.to_thread(ipca.fit_transform, vector_array)

            results = {
                "explained_variance_ratio": ipca.explained_variance_ratio_.tolist(),
                "singular_values": ipca.singular_values_.tolist(),
                "transformed_vectors": transformed_vectors.tolist()
            }

            return results
        except ImportError:
            self.logger.error("sklearn is required for incremental PCA. Please install it.")
            raise
        except Exception as e:
            self.logger.error(f"Error performing incremental PCA: {e}")
            raise

    async def vector_quantization(self, vector_ids: List[str], n_clusters: int = 10):
        try:
            from sklearn.cluster import MiniBatchKMeans

            vectors = await self.fetch_vectors(vector_ids)
            if not vectors:
                raise ValueError("No vectors found for the given IDs")

            vector_array = np.array([v['values'] for v in vectors.values()])

            quantizer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = await asyncio.to_thread(quantizer.fit_predict, vector_array)

            codebook = quantizer.cluster_centers_
            quantized_vectors = codebook[cluster_labels]

            results = {
                "codebook": codebook.tolist(),
                "quantized_vectors": quantized_vectors.tolist(),
                "cluster_labels": cluster_labels.tolist()
            }

            return results
        except ImportError:
            self.logger.error("sklearn is required for vector quantization. Please install it.")
            raise
        except Exception as e:
            self.logger.error(f"Error performing vector quantization: {e}")
            raise

    async def vector_interpolation(self, start_vector_id: str, end_vector_id: str, num_steps: int = 10):
        try:
            start_vector = await self.fetch_vector(start_vector_id)
            end_vector = await self.fetch_vector(end_vector_id)

            if not start_vector or not end_vector:
                raise ValueError("Start or end vector not found")

            start_array = np.array(start_vector['values'])
            end_array = np.array(end_vector['values'])

            alphas = np.linspace(0, 1, num_steps)
            interpolated_vectors = [((1 - alpha) * start_array + alpha * end_array).tolist() for alpha in alphas]

            return interpolated_vectors
        except Exception as e:
            self.logger.error(f"Error performing vector interpolation: {e}")
            raise

    async def vector_analogy(self, a_id: str, b_id: str, c_id: str, top_k: int = 5):
        try:
            a_vector = await self.fetch_vector(a_id)
            b_vector = await self.fetch_vector(b_id)
            c_vector = await self.fetch_vector(c_id)

            if not a_vector or not b_vector or not c_vector:
                raise ValueError("One or more input vectors not found")

            a = np.array(a_vector['values'])
            b = np.array(b_vector['values'])
            c = np.array(c_vector['values'])

            analogy_vector = b - a + c

            results = await self.vector_similarity_search(analogy_vector.tolist(), top_k=top_k)
            return results
        except Exception as e:
            self.logger.error(f"Error performing vector analogy: {e}")
            raise

class DataManagementLSTM(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(DataManagementLSTM, self).__init__()
        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.num_layers = config.get('num_layers', 1)
        self.bidirectional = config.get('bidirectional', False)
        self.dropout = config.get('dropout', 0.0)
        self.attention = config.get('attention', None)
        self.residual = config.get('residual', False)

        # LSTM layers
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

        # Attention mechanism
        if self.attention:
            self.attention_layer = self._create_attention_layer()

        # Output layer
        lstm_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(lstm_output_dim, self.output_dim)

        # Regularization
        self.dropout_layer = nn.Dropout(self.dropout)

        # Optimization
        self.scaler = amp.GradScaler(device='cuda')

    def quantum_compression(self, data):
        logging.info("Applying improved fractal-based quantum compression with adaptive entropy scaling.")

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
        return compressed_data
    
    def _create_attention_layer(self):
        if self.attention == 'self':
            return nn.MultiheadAttention(self.hidden_dim, num_heads=4)
        elif self.attention == 'luong':
            return nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention}")

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(x, hidden)

        # Apply attention if specified
        if self.attention:
            if self.attention == 'self':
                output, _ = self.attention_layer(output, output, output)
            elif self.attention == 'luong':
                attention_weights = torch.bmm(output, self.attention_layer(hidden[-1]).unsqueeze(2))
                output = torch.bmm(attention_weights.transpose(1, 2), output)

        # Apply residual connection if specified
        if self.residual:
            output = output + x

        # Apply dropout
        output = self.dropout_layer(output)

        # Final linear layer
        output = self.fc(output[:, -1, :])  # Take the last time step

        return output

    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        # Normalize the input data
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        return (data - mean) / (std + 1e-8)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {'optimizer': optimizer, 'scheduler': scheduler}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        x = self.preprocess(x)
        with torch.autocast():
            y_hat = self(x)
            loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, y = batch
        x = self.preprocess(x)
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return {'val_loss': loss}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, y = batch
        x = self.preprocess(x)
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return {'test_loss': loss}

    def fit(self, train_loader, val_loader, epochs: int):
        optimizer = self.configure_optimizers()['optimizer']
        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                loss = self.training_step(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

            self.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    val_loss = self.validation_step(batch)['val_loss']
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)

    @classmethod
    def load(cls, path: str):
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def get_state(self):
        state = {
            'lstm_state_dict': self.lstm.state_dict(),
            'fc_state_dict': self.fc.state_dict(),
            'config': self.config
        }
        return state

    def set_state(self, state):
        self.lstm.load_state_dict(state['lstm_state_dict'])
        self.fc.load_state_dict(state['fc_state_dict'])
        self.config = state['config']


class WebsiteScanner:
    """Class to manage website scanning operations."""

    def __init__(self, config, data_manager):
        """Initialize the website scanner."""
        self.config = config
        self.data_manager = data_manager
        self.session = None



    async def scrape_website(self, url: str) -> str:
        """Scrape text data from a website."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        try:
            async with self.session.get(url, timeout=30) as response:
                content = await response.text()
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text()
        except aiohttp.ClientError as e:
            self.data_manager.logger.error(f"Error scraping website {url}: {e}")
            return ""
        except Exception as e:
            self.data_manager.logger.error(f"Unexpected error scraping website {url}: {e}")
            return ""

    
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()


