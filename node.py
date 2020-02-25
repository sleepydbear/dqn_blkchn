from dqn import QNetwork,experienceReplayBuffer,DQNAgent
from blockchain import Blockchain
import gym


envs = ['Pendulum-v0','CartPole-v1','Acrobot-v1','MountainCar-v0','LunarLander-v2']

class Node:

    def __init__(self):
        self.wallet = Wallet()
        self.wallet.create_keys()
        self.blockchain = Blockchain(self.wallet.public_key)
        self.env = None
        self.buffer = None
        self.dqn = None
        self.agent = None
        
    def get_user_choice(self):
        """Prompts the user for its choice and return it."""
        user_input = input('Your choice: ')
        return user_input

    def listen_for_input(self):
        waiting_for_input = True

        while waiting_for_input:
            print('Please choose')
            print('1: Add a new environment value')
            print('2: Create agent')
            print('3: Train')
            print('q: Quit')


            user_choice = self.get_user_choice()
            if user_choice == '1':
                for i,e in enumerate(envs):
                    print('{}: {}'.format(i,e))
                selection = input('Please choose one of the above environments: ')
                self.env = gym.make(envs[int(selection)])
                print('selected {}'.format(envs[int(selection)]))
            elif user_choice == '2':
                if self.env != None:
                    self.buffer = experienceReplayBuffer(memory_size=1000000, burn_in=10000)
                    self.dqn =  QNetwork(self.env, learning_rate=1e-3)
                    self.agent = DQNAgent(self.env, self.dqn, self.buffer)
                    print('Agent created! Ready to train')
                else:
                    print('Environment empty! Set learning environment first')
            elif user_choice == '3':
                if self.env != None and self.agent != None:
                    print('Training started')
                    self.agent.train(max_episodes=20000, network_update_frequency=1, network_sync_frequency=2500)
                    print('Training complete')
                else:
                    print('check env or agent first')

            elif user_choice == 'q':
                # This will lead to the loop to exist because it's running condition becomes False
                waiting_for_input = False
            else:
                print('Input was invalid, please pick a value from the list!')
            

        print('Done!')

if __name__ == '__main__':
    node = Node()
    node.listen_for_input()



# buffer = experienceReplayBuffer(memory_size=1000000, burn_in=10000)
# dqn = QNetwork(env, learning_rate=1e-3)
# agent = DQNAgent(env, dqn, buffer)
# agent.train(max_episodes=20000, network_update_frequency=1, 
#             network_sync_frequency=2500)

