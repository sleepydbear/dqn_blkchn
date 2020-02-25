from functools import reduce
import hashlib as hl
import json

#custom imports
from block import Block
from interaction import Interaction
from hash_utils import hash_block
from wallet import Wallet
from verification import Verification


class Blockchain:

    def __init__(self,public_key,node_id):
        genesis_block = Block(0, '', [], 100, 0)
        self.chain = [genesis_block]
        self.public_key = public_key
        self.open_interactions = []
        self.node_id = node_id
        self.load_data()

    def add_interaction(self,sender,message,metadata,timestamp,signature):
        interaction = Interaction(sender,message,metadata,timestamp,signature)
        self.open_interactions.append(interaction)
        self.save_data()

    def mine_block(self):
        last_block = self.chain[-1]
        hashed_last_block = hash_block(last_block)

        open_interactions_copy = self.open_interactions
        valid_open_interactions = []
        for i in open_interactions_copy:
            if Wallet.verify_interaction(i):
                valid_open_interactions.append(i)
        self.open_interactions = valid_open_interactions

        proof = self.proof_of_work()

        block = Block(len(self.chain), hashed_last_block, self.open_interactions, proof)

        self.chain.append(block)
        self.open_interactions = []

        self.save_data()



    def load_data(self):
        """Initialize blockchain + open transactions data from a file."""
        try:
            with open('files/blockchains/blockchain-{}.txt'.format(self.node_id), mode='r') as f:

                file_content = f.readlines()

                blockchain = json.loads(file_content[0][:-1])
                # We need to convert  the loaded data because Transactions should use OrderedDict
                updated_blockchain = []
                for block in blockchain:
                    converted_tx = [Interaction(
                        tx['sender'], tx['message'], tx['metadata'], tx['timestamp'], tx['signature']) for tx in block['interactions']]
                    updated_block = Block(
                        block['index'], block['previous_hash'], converted_tx, block['proof'], block['timestamp'])
                    updated_blockchain.append(updated_block)
                self.chain = updated_blockchain
                open_interactions = json.loads(file_content[1])
                # We need to convert  the loaded data because Transactions should use OrderedDict
                updated_interactions = []
                for tx in open_interactions:
                    updated_interaction = Interaction(
                        tx['sender'], tx['message'], tx['metadata'], tx['timestamp'], tx['signature'])
                    updated_interactions.append(updated_interaction)
                self.open_interactions = updated_interactions

        except (IOError, IndexError):
            self.save_data()
        finally:
            print('Cleanup!')

    def save_data(self):
        """Save blockchain + open transactions snapshot to a file."""
        try:
            with open('files/blockchains/blockchain-{}.txt'.format(self.node_id), mode='w') as f:
                saveable_chain = [block.__dict__ for block in [Block(block_el.index, block_el.previous_hash, [
                    tx.__dict__ for tx in block_el.interactions], block_el.proof, block_el.timestamp) for block_el in self.chain]]
                f.write(json.dumps(saveable_chain))
                f.write('\n')
                saveable_tx = [tx.__dict__ for tx in self.open_interactions]
                f.write(json.dumps(saveable_tx))


        except IOError:
            print('Saving failed!')

    def proof_of_work(self):
        last_block = self.chain[-1]
        last_hash = hash_block(last_block)
        proof = 0
        # Try different PoW numbers and return the first valid one
        while not Verification.valid_proof(self.open_interactions, last_hash, proof):
            proof += 1
        return proof

    def add_block(self,block):
        interactions = [Interaction(
            tx['sender'], tx['message'], tx['metadata'], tx['timestamp'], tx['signature']) for tx in block['interactions']]

        converted_block = Block(
            block['index'], block['previous_hash'], interactions, block['proof'], block['timestamp'])

        self.chain.append(converted_block)
        self.open_interactions = []
        self.save_data()
        
