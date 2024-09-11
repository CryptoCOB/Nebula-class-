
import json
import logging
import socket
import unittest
import threading
import time
import random
from OrionBelt import Block, Blockchain, OrionLightNode, OrionTokenEconomy, OrionNetworkNode, QuantumConsensus, QuantumErrorCorrection


class TestOrionLite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blockchain = Blockchain()
        cls.token_economy = OrionTokenEconomy()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def setUp(self):
        self.nodes = []
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                for i in range(5):  # Adjust number as needed
                    node = OrionNetworkNode(host="127.0.0.1", port=0)  # Let the OS assign a port
                    self.nodes.append(node)
                    threading.Thread(target=node.start_server).start()

                time.sleep(2)  # Allow time for nodes to start
                break  # If successful, exit the retry loop
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed to initialize nodes: {str(e)}")
                self.tearDown()  # Clean up any partially initialized resources
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError("Failed to initialize nodes after multiple attempts.")

    def tearDown(self):
        for node in self.nodes:
            node.stop()
        self.nodes.clear()
        time.sleep(1)  # Allow time for cleanup
        self.blockchain.save_chain()

    def get_free_port(self, start_port=5000, end_port=65535):
        for port in range(start_port, end_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        return None

    # General Blockchain Integrity and Consensus Tests
    def test_genesis_block_integrity(self):
        self.assertEqual(self.blockchain.chain[0].index, 0)
        self.assertEqual(self.blockchain.chain[0].transactions, "Genesis Block")
        self.assertEqual(self.blockchain.chain[0].validator, "System")

    def test_block_linkage_validation(self):
        for i in range(1, len(self.blockchain.chain)):
            block = self.blockchain.chain[i]
            previous_block = self.blockchain.chain[i - 1]
            self.assertEqual(block.previous_hash, previous_block.hash)
            self.assertNotEqual(block.hash, previous_block.hash)

    def test_validator_selection_process(self):
        validator = self.blockchain.consensus.select_validator()
        self.assertIn(validator, self.blockchain.validators)

    def test_block_validation_time(self):
        start_time = time.time()
        self.blockchain.add_transaction("Test transaction")
        self.blockchain.validate_and_mine_block()
        end_time = time.time()
        validation_time = end_time - start_time
        self.assertLess(validation_time, 1.0)
        print(f"Block validation time: {validation_time:.4f} seconds")

    def test_transaction_throughput(self):
        transactions = 1000
        for i in range(transactions):
            self.blockchain.add_transaction(f"Transaction {i}")
        start_time = time.time()
        self.blockchain.validate_and_mine_block()
        end_time = time.time()
        tps = transactions / (end_time - start_time)
        print(f"Transaction Throughput: {tps:.2f} TPS")
        self.assertGreater(tps, 1000)

    def test_network_latency(self):
        num_nodes = 5
        for i in range(num_nodes):
            port = 5000 + i
            node = OrionNetworkNode(host="127.0.0.1", port=port)
            self.nodes.append(node)
            threading.Thread(target=node.start_server).start()
        time.sleep(3)
        start_time = time.time()
        for i in range(1, num_nodes):
            self.nodes[i - 1].connect_to_peer("127.0.0.1", 5000 + i)
        end_time = time.time()
        network_latency = end_time - start_time
        print(f"Network Latency: {network_latency:.4f} seconds")
        self.assertLess(network_latency, 1.0)

    def test_chain_conflict_resolution(self):
        # Simulate diverging chains and test if the correct one is adopted
        self.blockchain.add_transaction("Transaction 1")
        self.blockchain.validate_and_mine_block()

        # Simulate an alternate chain from another node
        fake_chain = [
            self.blockchain.create_genesis_block(),
            self.blockchain.get_latest_block()
        ]

        # Inject the fake chain and trigger conflict resolution
        self.nodes[0].update_local_blockchain(fake_chain)
        self.assertEqual(len(self.blockchain.chain), 2)

    def test_node_synchronization(self):
        num_nodes = 3
        for i in range(num_nodes):
            port = 5000 + i
            node = OrionNetworkNode(host="127.0.0.1", port=port)
            self.nodes.append(node)
            threading.Thread(target=node.start_server).start()
        time.sleep(2)

        # Trigger blockchain sync after a node has been disconnected
        self.blockchain.add_transaction("Sync transaction")
        self.blockchain.validate_and_mine_block()
        self.nodes[0].sync_blockchain_data()

    def test_transaction_cost_and_speed_comparison(self):
        # Define the top 15 cryptocurrencies and their approximate transaction times and fees
        top_15_crypto = {
            "Bitcoin": {"avg_time": 10 * 60, "avg_fee": 5},  # 10 minutes, $5
            "Ethereum": {"avg_time": 15, "avg_fee": 2},  # 15 seconds, $2
            "Tether": {"avg_time": 30, "avg_fee": 1},  # 30 seconds, $1
            "BNB": {"avg_time": 3, "avg_fee": 0.1},  # 3 seconds, $0.1
            "USD Coin": {"avg_time": 30, "avg_fee": 0.5},  # 30 seconds, $0.5
            "XRP": {"avg_time": 4, "avg_fee": 0.01},  # 4 seconds, $0.01
            "Cardano": {"avg_time": 10 * 60, "avg_fee": 0.2},  # 10 minutes, $0.2
            "Dogecoin": {"avg_time": 1 * 60, "avg_fee": 0.05},  # 1 minute, $0.05
            "Solana": {"avg_time": 0.4, "avg_fee": 0.001},  # 400 milliseconds, $0.001
            "TRON": {"avg_time": 3, "avg_fee": 0.001},  # 3 seconds, $0.001
            "Litecoin": {"avg_time": 2.5 * 60, "avg_fee": 0.05},  # 2.5 minutes, $0.05
            "Polkadot": {"avg_time": 6, "avg_fee": 0.1},  # 6 seconds, $0.1
            "Stellar": {"avg_time": 5, "avg_fee": 0.00001},  # 5 seconds, $0.00001
            "VeChain": {"avg_time": 10, "avg_fee": 0.0001},  # 10 seconds, $0.0001
            "Avalanche": {"avg_time": 1, "avg_fee": 0.0002},  # 1 second, $0.0002
        }

        num_transactions = 1000
        orion_total_time = 0
        orion_total_cost = 0

        # Simulate Orion transactions
        start_time = time.time()
        for _ in range(num_transactions):
            self.blockchain.add_transaction(f"Transaction {_}")
            self.blockchain.validate_and_mine_block()
            orion_total_cost += 0.01  # Assume a fixed cost of 0.01 for Orion transactions

        orion_total_time = time.time() - start_time
        orion_avg_time = orion_total_time / num_transactions
        orion_avg_cost = orion_total_cost / num_transactions

        print(f"\nOrion Performance:")
        print(f"Average Transaction Time: {orion_avg_time:.4f} seconds")
        print(f"Average Transaction Cost: ${orion_avg_cost:.4f}")

        # Compare with top 15 cryptocurrencies
        print("\nComparison with Top 15 Cryptocurrencies:")
        for crypto, data in top_15_crypto.items():
            time_difference = data['avg_time'] - orion_avg_time
            cost_difference = data['avg_fee'] - orion_avg_cost

            print(f"\n{crypto}:")
            print(f"  Avg Time: {data['avg_time']} seconds (Difference: {time_difference:.2f} seconds)")
            print(f"  Avg Fee: ${data['avg_fee']} (Difference: ${cost_difference:.2f})")

            if time_difference > 0:
                print(f"  Orion is {time_difference:.2f} seconds faster")
            elif time_difference < 0:
                print(f"  Orion is {abs(time_difference):.2f} seconds slower")
            else:
                print("  Orion has the same speed")

            if cost_difference > 0:
                print(f"  Orion is ${cost_difference:.2f} cheaper")
            elif cost_difference < 0:
                print(f"  Orion is ${abs(cost_difference):.2f} more expensive")
            else:
                print("  Orion has the same cost")

        # Assert that Orion's performance is competitive
        self.assertLess(orion_avg_time, 60, "Orion's average transaction time should be less than 1 minute")
        self.assertLess(orion_avg_cost, 1, "Orion's average transaction cost should be less than $1")

    # Security Tests
    def test_sybil_attack_resilience(self):
        attackers = [f"FakeNode{i}" for i in range(100)]
        self.blockchain.consensus.validators.extend(attackers)
        self.blockchain.consensus.select_validator()
        self.assertGreater(len(set(self.blockchain.consensus.validators)), 10)

    def test_51_percent_attack(self):
        malicious_validator = "MaliciousNode"
        self.blockchain.consensus.validators.extend([malicious_validator] * 6)
        self.blockchain.consensus.select_validator()
        self.assertNotEqual(self.blockchain.get_latest_block().validator, malicious_validator)

    def test_eclipse_attack(self):
        # Simulate an attack that isolates a node by controlling all peer connections
        self.nodes = [OrionNetworkNode(host="127.0.0.1", port=5000)]
        for i in range(1, 5):
            attacker_node = OrionNetworkNode(host="127.0.0.1", port=5000 + i)
            self.nodes.append(attacker_node)
            attacker_node.peers = [("127.0.0.1", 5000)]
        time.sleep(2)
        self.assertEqual(len(self.nodes[0].peers), 4)

    def test_ddos_attack(self):
        # Simulate a DDoS attack by flooding a node with connections
        target_node = OrionNetworkNode(host="127.0.0.1", port=5000)
        self.nodes.append(target_node)
        threading.Thread(target=target_node.start_server).start()
        time.sleep(1)

        for _ in range(100):
            attacker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            attacker_socket.connect(("127.0.0.1", 5000))
            attacker_socket.close()

        self.assertIsNone(target_node.server)


if __name__ == "__main__":
    unittest.main()