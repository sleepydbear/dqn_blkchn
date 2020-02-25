from time import time


class Block():

    def __init__(self, index, previous_hash, interactions, proof, time=time()):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = time
        self.interactions = interactions
        self.proof = proof

    def __repr__(self):
        return str(self.__dict__)
