from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA256
import Crypto.Random
import binascii

import base64
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet


class Wallet:
    """Creates, loads and holds private and public keys. Manages transaction signing and verification."""

    def __init__(self, node_id):
        self.private_key = None
        self.public_key = None
        self.encryption_password = None
        self.node_id = node_id
        self.ferent_key = None

    def create_keys(self):
        """Create a new pair of private and public keys."""
        private_key, public_key = self.generate_keys()
        self.private_key = private_key
        self.public_key = public_key

        password = private_key.encode() 
        salt = b'' 
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend())
        key = base64.urlsafe_b64encode(kdf.derive(password))

        self.encryption_password = key.decode()
        self.ferent_key = Fernet(self.encryption_password.encode())



    def save_keys(self):
        """Saves the keys to a file (wallet.txt)."""
        if self.public_key != None and self.private_key != None:
            try:
                with open('files/wallets/wallet-{}.txt'.format(self.node_id), mode='w') as f:
                    f.write(self.public_key)
                    f.write('\n')
                    f.write(self.private_key)
                    f.write('\n')
                    f.write(self.encryption_password)
                return True
            except (IOError, IndexError):
                print('Saving wallet failed...')
                return False

    def load_keys(self):
        """Loads the keys from the wallet.txt file into memory."""
        try:
            with open('files/wallets/wallet-{}.txt'.format(self.node_id), mode='r') as f:
                keys = f.readlines()
                public_key = keys[0][:-1]
                private_key = keys[1][:-1]
                encryption_password = keys[2]
                self.public_key = public_key
                self.private_key = private_key
                self.encryption_password = encryption_password
                self.ferent_key = Fernet(encryption_password.encode())
                

            return True
        except (IOError, IndexError):
            print('Loading wallet failed...')
            return False

    def generate_keys(self):
        """Generate a new pair of private and public key."""
        private_key = RSA.generate(1024, Crypto.Random.new().read)
        public_key = private_key.publickey()
        return (binascii.hexlify(private_key.exportKey(format='DER')).decode('ascii'), binascii.hexlify(public_key.exportKey(format='DER')).decode('ascii'))

    def sign_interaction(self, sender, message, metadata, timestamp):

        signer = PKCS1_v1_5.new(RSA.importKey(binascii.unhexlify(self.private_key)))
        h = SHA256.new((str(sender) + str(message) + str(metadata) + str(timestamp)).encode('utf8'))
        signature = signer.sign(h)
        return binascii.hexlify(signature).decode('ascii')

    def encrypt_data(self,data):
        msg = self.ferent_key.encrypt(data.encode()).decode()
        return msg

    def decrypt_data(self,data):
        return self.ferent_key.decrypt(data.encode()).decode()

    def __repr__(self):
        return str(self.__dict__)

    @staticmethod
    def verify_interaction(interaction):

        public_key = RSA.importKey(binascii.unhexlify(interaction.sender))
        verifier = PKCS1_v1_5.new(public_key)
        h = SHA256.new((str(interaction.sender) + str(interaction.message) + str(interaction.metadata) + str(interaction.timestamp)).encode('utf8'))
        return verifier.verify(h, binascii.unhexlify(interaction.signature))