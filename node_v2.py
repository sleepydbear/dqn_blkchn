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
from hash_utils import hash_string_256
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
peers = 6

test_dqn = None
test_agent = None
test_buffer =None
test_env = None
best_policy_performances = {}

@app.route('/refresh', methods=['PUT'])
def refresh_history():
    blk_chn = blockchain.chain.copy()
    outcomes = []
    for b in blk_chn:
        outcomes += b.outcome


    response = {
            'message': 'refresh request received',
            'outcomes': outcomes
        }
    return jsonify(response), 201

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
    global test_agent,test_buffer,test_dqn,best_policy_performances
    values = request.get_json()

    sender = values['sender']
    message = values['message']
    metadata = values['metadata']
    signature = values['signature']
    timestamp = values['timestamp']
    blockchain.add_interaction(sender,message,metadata,timestamp,signature)
    print(best_policy_performances)
    if message == 'NN update':
        test_env_name = metadata['env']
        test_env = gym.make(test_env_name)
        test_buffer = experienceReplayBuffer(memory_size=1000000, burn_in=10000)
        test_dqn =  QNetwork(test_env, learning_rate=1e-3)
        test_agent = DQNAgent(blockchain, wallet, port, peers, test_env, test_dqn, test_buffer)
        pickled_weights = metadata['weights']
        
        if test_env_name in best_policy_performances:
            cp = best_policy_performances[test_env_name]
            result = test_agent.test(pickled_weights)
            impr = (result - cp)/cp
            print(result,cp,impr)
            if impr > 0.25:
                reply = 'Improvement observed'
                best_policy_performances[test_env_name] = result
            else:   
                reply = 'Dud policy'
        else:
            reply = 'Policy tracking started'
            result = test_agent.test(pickled_weights)
            best_policy_performances[test_env_name] = result

        #broadcast transaction reply
        hashed_signature = hash_string_256(signature.encode())
        metadata = {'sig_hash': hashed_signature,'sender':sender,'env': metadata['env']}
        message = reply
        timestamp = time.time()
        signature = wallet.sign_interaction(wallet.public_key,message,metadata,timestamp)
        blockchain.add_interaction(wallet.public_key,message,metadata,timestamp,signature)
        for peer in range(5000,5000+peers+1):
            if peer != port:
                url = 'http://192.168.1.9:{}/broadcast-transaction'.format(str(peer))
                try:
                    post_transaction = requests.put(url, json={'sender': wallet.public_key, 'message': message,
                                                'metadata':metadata, 'timestamp':timestamp, 'signature': signature})                
                except requests.exceptions.ConnectionError:
                    continue

    response = {
            'message': 'request received'
        }
    return jsonify(response), 201


@app.route('/train', methods=['PUT'])
def train():
    global env,buffer,dqn,agent,assigned
    values = request.get_data()
    # myJson = json.loads(values.decode("utf-8"))

    if assigned:
        agent.train(max_episodes=1000, network_update_frequency=1, network_sync_frequency=2500)
        response = {
                'message': 'Training done'
            }
        return jsonify(response), 201
    else:
        response = {
                'message': 'Task assignment not done, cannot train!'
            }
        return jsonify(response), 500

@app.route('/broadcast-block', methods=['PUT'])
def broadcast_block():
    values = request.get_data()
    myJson = json.loads(values.decode("utf-8"))
    block = myJson['block']
    blockchain.add_block(block)
    response = {
            'message': 'request received'
        }
    return jsonify(response), 201


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

    blockchain = Blockchain(wallet.public_key,peers,port)

    app.run(host='192.168.1.9', port=int(port))