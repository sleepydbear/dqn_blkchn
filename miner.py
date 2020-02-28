from blockchain import Blockchain
from wallet import Wallet
from verification import Verification
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import requests
import time,threading

message = 'hello world'
metadata = 'vehicle info: 151223'

app = Flask(__name__)
CORS(app)

mining_schedule_delay = 5

peers = 6


@app.route('/broadcast-transaction', methods=['PUT'])
def broadcast_transaction():
    values = request.get_json()

    sender = values['sender']
    message = values['message']
    metadata = values['metadata']
    signature = values['signature']
    timestamp = values['timestamp']
    blockchain.add_interaction(sender,message,metadata,timestamp,signature)
    response = {
            'message': 'request received'
        }
    return jsonify(response), 201



def schedule_mining(thread_name,delay):
    t1 = time.time()
    while True:
        ct = time.time()
        diff = ct - t1
        if diff > delay:
            t1 = time.time()
            print('mining1 from {}'.format(thread_name))

            # print(blockchain.open_interactions)
            blockchain.mine_block()
            block = blockchain.chain[-1]
            last_block = block.__dict__.copy()
            last_block['interactions'] = [
                tx.__dict__ for tx in last_block['interactions']]
            print(last_block)


            for peer in range(5000,5000+peers+1):
                if peer != port:
                    url = 'http://192.168.1.9:{}/broadcast-block'.format(str(peer))
                    try:
                        post_transaction = requests.put(url, json={'block': last_block})                
                    except requests.exceptions.ConnectionError:
                        continue

def start_app():
    app.run(host='192.168.1.9', port=int(port))

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

    thread1 = threading.Thread(target=schedule_mining,args=('t1',mining_schedule_delay,))
    thread2 = threading.Thread(target=start_app,args=())

    thread1.start()
    thread2.start()

    print('Threads started!')

