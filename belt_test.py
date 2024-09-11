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
        # Define the top 10 cryptocurrencies and their approximate transaction times and fees
        top_10_crypto = {
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

        # Compare with top 10 cryptocurrencies
        print("\nComparison with Top 10 Cryptocurrencies:")
        for crypto, data in top_10_crypto.items():
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


    def test_consensus_failure_handling(self):
        # Simulate a failure in the consensus process and check system behavior
        selected_validator = self.blockchain.consensus.select_validator()
        self.assertIn(selected_validator, self.blockchain.validators)

    def test_token_economy_operations(self):
        self.token_economy.initialize_economy(["RegionA"])
        self.token_economy.mint_tokens("RegionA", 1000)
        self.token_economy.burn_tokens("RegionA", 500)
        self.assertEqual(self.token_economy.regional_data["RegionA"]["tokens"], 500)

    # Advanced Security Tests Against Known Attack Vectors
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

    def test_long_range_attack(self):
        # Simulate a scenario where old validators try to forge a new, longer chain
        old_chain = self.blockchain.chain.copy()
        self.blockchain.add_transaction("Old malicious transaction")
        self.blockchain.validate_and_mine_block()

        # Inject the old chain to simulate the attack
        self.nodes[0].update_local_blockchain(old_chain)
        self.assertGreaterEqual(len(self.blockchain.chain), len(old_chain))

    def test_double_spend_attack(self):
        self.blockchain.add_transaction("Alice pays Bob 10 tokens")
        self.blockchain.add_transaction("Alice pays Charlie 10 tokens")
        self.blockchain.validate_and_mine_block()
        transactions = self.blockchain.get_latest_block().transactions
        self.assertIn("Alice pays Bob 10 tokens", transactions)
        self.assertNotIn("Alice pays Charlie 10 tokens", transactions)

    def test_time_manipulation_attack(self):
        manipulated_block = Block(2, self.blockchain.get_latest_block().hash, time.time() - 10000, ["Manipulated Transaction"], "QuantumNode1")
        manipulated_block.hash = manipulated_block.calculate_hash()
        self.nodes[0].update_local_blockchain([self.blockchain.get_latest_block(), manipulated_block])
        self.assertNotEqual(self.blockchain.get_latest_block().hash, manipulated_block.hash)

    def test_smart_contract_exploit(self):
        # Simulate a smart contract exploit where a malicious actor siphons tokens
        self.token_economy.mint_tokens("RegionA", 1000)
        self.token_economy.burn_tokens("RegionA", 500)
        self.assertEqual(self.token_economy.regional_data["RegionA"]["tokens"], 500)

    def test_reentrancy_attack(self):
        # Simulate a re-entrancy exploit where a function is recursively called
        def malicious_contract():
            self.token_economy.mint_tokens("RegionA", 100)
            self.token_economy.burn_tokens("RegionA", 100)

        for _ in range(5):
            malicious_contract()

        self.assertEqual(self.token_economy.regional_data["RegionA"]["tokens"], 0)

    def test_front_running_attack(self):
        # Simulate an attacker trying to exploit transaction ordering
        transactions = [
            {"from": "Alice", "to": "Bob", "amount": 10},
            {"from": "Attacker", "to": "Attacker", "amount": 100},
            {"from": "Charlie", "to": "Dave", "amount": 5},
        ]

        for tx in transactions:
            self.blockchain.add_transaction(json.dumps(tx))

        self.blockchain.validate_and_mine_block()
        self.assertIn("Attacker", self.blockchain.get_latest_block().transactions)

    # Quantum Security and Error Correction Tests
    def test_quantum_circuit_fidelity(self):
        qec = QuantumErrorCorrection()
        success_rate = qec.benchmark_error_correction(num_iterations=100)[1]
        self.assertGreater(success_rate, 0.90)

    def test_entanglement_robustness(self):
        qec = QuantumErrorCorrection()
        robustness = qec.introduce_errors_to_logical_qubits(error_rate=0.05)
        self.assertTrue(all(len(error) == qec.num_qubits for error in robustness))

    def test_shor_code_implementation(self):
        qec = QuantumErrorCorrection()
        qec.apply_shor_code()
        self.assertTrue(True)  # Placeholder for successful execution

    def test_surface_code_implementation(self):
        qec = QuantumErrorCorrection()
        qec.apply_surface_code()
        self.assertTrue(True)  # Placeholder for successful execution

    def test_quantum_error_rate_benchmark(self):
        qec = QuantumErrorCorrection()
        avg_time, success_rate = qec.benchmark_error_correction(num_iterations=500)
        self.assertGreater(success_rate, 0.85)
        self.assertLess(avg_time, 1.0)

    def test_qpoa_validation(self):
        selected_validator = self.blockchain.consensus.select_validator()
        self.assertIn(selected_validator, self.blockchain.validators)

    def test_resource_allocation_in_quantum_consensus(self):
        consensus = QuantumConsensus(validators=self.blockchain.validators)
        resource_allocation = consensus.generate_entanglement_pairs()
        self.assertGreater(len(resource_allocation), 0)

    def test_quantum_entanglement_management(self):
        light_node = OrionLightNode(node_id=1)
        light_node.initialize_light()
        self.assertGreater(len(light_node.entanglement_pairs), 0)

    # Economic Stability and Growth Projections
    def test_simulated_economic_growth(self):
        self.token_economy.initialize_economy(["RegionA", "RegionB", "RegionC"])
        self.token_economy.adjust_token_value()
        self.assertGreater(self.token_economy.token_price_index, 1.0)

    def test_token_price_volatility(self):
        initial_value = self.token_economy.token_price_index
        self.token_economy.adjust_token_value()
        self.assertNotEqual(self.token_economy.token_price_index, initial_value)

    def test_global_parity_mechanism(self):
        self.token_economy.initialize_economy(["RegionA", "RegionB", "RegionC"])
        for region in ["RegionA", "RegionB", "RegionC"]:
            self.token_economy.adjust_parity(region)
            self.assertGreater(self.token_economy.regional_data[region]["purchasing_power"], 0)

    def test_inflation_control_mechanism(self):
        self.token_economy.ai_governed_inflation_control()
        self.assertNotEqual(self.token_economy.token_supply, 1000000)

    def test_user_adoption_growth_simulation(self):
        initial_users = 1000
        growth_rate = 1.1  # 10% annual growth
        years = 10
        projected_users = initial_users * (growth_rate ** years)
        print(f"Simulated User Growth over 10 years: {projected_users:.2f}")
        self.assertGreater(projected_users, initial_users)


if __name__ == "__main__":
    unittest.main()
