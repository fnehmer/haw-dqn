import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os # for creating directories

env = gym.make('CartPole-v0') # initialise environment

state_size = env.observation_space.shape[0] # 4 [location of cart, velocity of cart, angle of pole, angular velocity of pole]

action_size = env.action_space.n # 2 [links, rechts]

batch_size = 32

n_episodes = 0

output_dir = 'model/current_training/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)    # Double Ended Queue: Eine Liste an der man vorne und hinten anfügen kann
        self.gamma = 0.95                   # Gamma: Discount Factor: Bestimmt einen Discount auf Rewards die weiter weg in der Zukunft liegen, da diese eine geringere Aussagekraft haben könnten
        self.epsilon = 1.0                  # Exploration Rate: Die Rate, in der zufällige Aktionen gelernt werden
        self.epsilon_decay = 0.995          # Die Rate in die Explorationsrate verkleinert wird. Später soll der Agent weniger Explorieren
        self.epsilon_min = 0.01             # Es soll immernoch eine minimale Explorationsrate geben
        self.learning_rate = 0.001          # Die Rate in der das NN Modellparameter anpasst mithilfe von 'Stochastic gradient descent' um Kosten zu reduzieren
        self.model = self._build_model()    # private method 
    
    def _build_model(self):
        '''Neuronales Netzwerk um Q*(s,a) zu approximieren
        '''
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) # 1st hidden layer; states as input
        model.add(Dense(24, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear')) # 2 actions, so 2 output neurons: 0 and 1 (L/R)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def construct_rewards_mean_list(self, rewards, step_size):
        '''Erstellt eine List, die in 'step_size'er Schritten die durchschnittlichen
        Rewards aus der Liste der aller Rewards errechnet. Wird für das Plotten benutzt
        '''
        tmp_val = 0
        output = []

        for i, reward in enumerate(rewards):
            tmp_val += reward
            if ((i+1) % step_size) == 0:
                output.append(tmp_val / step_size)
                tmp_val = 0
        return output

    # "weights_0950.hdf5"
    def play(self, input_dir):
        '''Spielt und visualiert das Spiel 3-Mal (ohne weiteres Training)'''
        input("Spiel starten?")
        self.load(input_dir)
        for playthrough in range(3):
            play_state = env.reset()
            done = False
            total_reward = 0
            while not done:
                env.render()
                play_state = np.reshape(play_state, [1, state_size])
                action = np.argmax(self.model.predict(play_state))
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                play_state = next_state
            print("Playthrough No. " + str(playthrough + 1) + "; Total Reward: " + str(total_reward))
    
    def remember(self, state, action, reward, next_state, done):
        '''Liste der vorherigen Spiele erweitern, wird beim Re-Training benutzt'''
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        '''Gibt eine Aktion wieder, entweder zufällig oder Prediction, hängt von epsilon-greedy ab'''
        if np.random.rand() <= self.epsilon: # 'Zufällige' Aktion
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) # Aktion basierend auf Modelprediction
        return np.argmax(act_values[0]) # Nehme die Aktion mit dem höchsten erwarteten Reward

    def replay(self, batch_size):
        '''Trainiert das NN mit der Hilfe des Memory'''
        minibatch = random.sample(self.memory, batch_size) # Nehme ein "Minibatch" aus dem memory
        for state, action, reward, next_state, done in minibatch: # Betrachte jeden Eintrag aus minibatch
            target = reward # Wenn das Spiel vorbei ist, ist es kein Geheimnis was der Reward ist
            if not done: # Wenn das Spiel nicht vorbei ist, müssen wir schätzen, wie groß unser erwarteter zukünftiger Reward ist
                target = (reward + self.gamma * # (target) = aktuellerreward + (discount rate gamma) * 
                          np.amax(self.model.predict(next_state)[0])) # (maximaler zukünftiger Reward)
            target_f = self.model.predict(state) # den ungefähren zukünftigen reward auf den aktuellen state mappen
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)# x=state, y=target_f Nur eine Epoche, weil wir nur ein Eintrag im Memory trainieren
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # Epsilon abbauen, um die Exploitation Rate zu erhöhen

    def load(self, name):
        '''Gewichte eines übergebenen NN laden'''
        self.model.load_weights(name)

    def save(self, name):
        '''Aktuelle Gewichte des NN speichern'''
        self.model.save_weights(name)

if __name__ == "__main__":
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    rewards = [] # fürs plotten

    done = False
    for e in range(n_episodes): # Über die Anzahl der Episoden des Spiels iterieren
        state = env.reset() # Zu jeder Epsiode muss von vorne angefangen werden
        state = np.reshape(state, [1, state_size])
        
        for time in range(201):  # time ist ein Frame im Training
            # env.render()
            # Memory auffüllen uns spielen
            action = agent.get_action(state) # Aktion bestimmen (entweder zufällig oder mit dem Modell)
            next_state, reward, done, _ = env.step(action) # Aktion ausführung und Observation erhalten       
            reward = reward if not done else -10 # Reward erhöhen wenn Spiel noch läuft ansonsten Strafe von -10 als Reward      
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done) # Aktuelle Aktion mit Observation und nächstem state abspeichern        
            state = next_state # state auf den ausgeführten state setzen       
            if done: # Episode ist vorbei, wenn die Pole umkippt oder 200 Frames erfolgreich waren
                rewards.append(time) # fürs plotten
                print("episode: {}/{}, score: {}, e: {:.2}" # Ausgabe für Episode und Erreichter Frame
                    .format(e, n_episodes, time, agent.epsilon))
                break # Schleife verlassen

        # Training durchführen
        if len(agent.memory) > batch_size:
            agent.replay(batch_size) # Replay durchführen, wenn das Memory groß genug ist
        if e % 10 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
    

    reward_means = agent.construct_rewards_mean_list(rewards=rewards, step_size=10) # fürs plotten

    # Visuelle Darstellung des Trainierten Agenten
    agent.play("model/good_model.hdf5")
    # agent.play("model/current_training/weights_0990.hdf5")

    # 
    plt.plot(range(len(reward_means)), reward_means, color="red")
    plt.show()