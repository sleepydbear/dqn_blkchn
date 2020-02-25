from blockchain import Blockchain
from block import Block
from wallet import Wallet
from verification import Verification
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import requests
from interaction import Interaction
import time

from dqn import QNetwork,experienceReplayBuffer,DQNAgent
import gym




message = 'hello world'
metadata = 'empty'

app = Flask(__name__)
CORS(app)

envs = ['Pendulum-v0','CartPole-v1','Acrobot-v1','MountainCar-v0','LunarLander-v2']

env = None
buffer = None
dqn = None
agent = None
assigned = False
peers = 2

test_dqn = None
test_agent = None
test_buffer =None
test_env = None

@app.route('/assign', methods=['PUT'])
def assign_task():
    global env,buffer,dqn,agent,assigned
    values = request.get_data()
    myJson = json.loads(values.decode("utf-8"))

    if myJson['message'] == 'Assignment':
        env_selection = myJson['data']['env']
        assigned = True
        env = gym.make(env_selection)
        buffer = experienceReplayBuffer(memory_size=1000000, burn_in=10000)
        dqn =  QNetwork(env, learning_rate=1e-3)
        agent = DQNAgent(blockchain, wallet, port, peers, env, dqn, buffer)

        metadata = myJson
        timestamp = time.time()
        message = 'Authorized to learn task'
        signature = wallet.sign_interaction(wallet.public_key,message,metadata,timestamp)
        blockchain.add_interaction(wallet.public_key,message,metadata,timestamp,signature)

    #broadcast transactions
    for peer in range(5000,5000+peers+1):
        if peer != port:
            url = 'http://192.168.1.9:{}/broadcast-transaction'.format(str(peer))
            try:
                post_transaction = requests.put(url, json={'sender': wallet.public_key, 'message': message,
                                            'metadata':metadata, 'timestamp':timestamp, 'signature': signature})                
            except requests.exceptions.ConnectionError:
                continue

    response = {
            'message': 'Task Assignment received, ready to train!'
        }
    return jsonify(response), 201

@app.route('/broadcast-transaction', methods=['PUT'])
def broadcast_transaction():
    global test_agent,test_buffer,test_dqn
    values = request.get_json()

    sender = values['sender']
    message = values['message']
    metadata = values['metadata']
    signature = values['signature']
    timestamp = values['timestamp']
    blockchain.add_interaction(sender,message,metadata,timestamp,signature)

    if message == 'NN update':
        test_env_name = metadata['env']
        test_env = gym.make(test_env_name)
        test_buffer = experienceReplayBuffer(memory_size=1000000, burn_in=10000)
        test_dqn =  QNetwork(test_env, learning_rate=1e-3)
        test_agent = DQNAgent(blockchain, wallet, port, peers, test_env, test_dqn, test_buffer)
        pickled_weights = metadata['weights']
        test_agent.test(pickled_weights)

    response = {
            'message': 'request received'
        }
    return jsonify(response), 201

@app.route('/baa', methods=['PUT'])
def baa():
    global test_agent,test_buffer,test_dqn
    
    test_buffer = experienceReplayBuffer(memory_size=1000000, burn_in=10000)
    test_dqn =  QNetwork(env, learning_rate=1e-3)
    test_agent = DQNAgent(blockchain, wallet, port, peers, env, test_dqn, test_buffer)
    
    test_agent.test()

    test_buffer = None
    test_dqn = None
    test_agent = None
    response = {
            'message': 'baa received'
        }
    return jsonify(response), 201

@app.route('/train', methods=['PUT'])
def train():
    global env,buffer,dqn,agent,assigned
    # values = request.get_data()
    # myJson = json.loads(values.decode("utf-8"))

    if assigned:
        agent.train(max_episodes=20000, network_update_frequency=1, network_sync_frequency=2500)
        response = {
                'message': 'Training done'
            }
        return jsonify(response), 201
    else:
        response = {
                'message': 'Task assignment not done, cannot train!'
            }
        return jsonify(response), 500

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=5000)
    args = parser.parse_args()
    port = args.port
    wallet = Wallet(port)
    #load wallet
    if not wallet.load_keys():
        wallet.create_keys()
        wallet.save_keys()
        print('new wallet created')
    else:
        print('loaded existing wallet')

    blockchain = Blockchain(wallet.public_key,port)

    app.run(host='192.168.1.9', port=int(port))