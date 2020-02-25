from hash_utils import hash_string_256, hash_block
from wallet import Wallet

class Verification:

    @staticmethod
    def valid_proof(interactions, last_hash, proof):

        # Create a string with all the hash inputs
        guess = (str([tx.to_ordered_dict() for tx in interactions]) + str(last_hash) + str(proof)).encode()

        guess_hash = hash_string_256(guess)


        return guess_hash[0:2] == '00'

    @classmethod
    def verify_chain(cls, blockchain):
        """ Verify the current blockchain and return True if it's valid, False otherwise."""
        for (index, block) in enumerate(blockchain):
            if index == 0:
                continue
            if block.previous_hash != hash_block(blockchain[index - 1]):
                return False
            if not cls.valid_proof(block.interactions, block.previous_hash, block.proof):
                return False
        return True

