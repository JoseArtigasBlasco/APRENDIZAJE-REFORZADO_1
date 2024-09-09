# El objetivo del agente es aprender a equilibrar un palo sobre un carrito moviéndose hacia la izquierda o la derecha.


import os
import gym
import numpy as np
from collections import deque
import random
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Configuramos entorno y agente
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]  #  determina el tamaño del espacio de observación
action_size = env.action_space.n # número de acciones posibles que el agente puede tomar en el entorno
batch_size = 32
n_episodes = 1000 # define cuántos episodios de entrenamiento se ejecutarán.
output_dir = 'model_output/cartpole/'

# Si no existe el directorio, se crea
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Definimos modelo red neuronal
def build_model():
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
    return model


# La clase DQNAgent define el agente que aprende a través de DQN:
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   # Determina la importancia de las recompensas futuras.
                            # Un valor cercano a 0 hace que el agente priorice las recompensas inmediatas,
                            # mientras que un valor cercano a 1 hace que considere las recompensas futuras.
        self.epsilon = 1.0  # Tasa de exploración inicial. Controla la probabilidad de que el agente elija
                            # una acción aleatoria (exploración) en lugar de la acción que actualmente cree
                            # que es la mejor (explotación).
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = build_model()
        self.target_model = build_model()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# Entrenamiento del agente
agent = DQNAgent(state_size, action_size)
done = False

for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_model()
            print(f"episode: {e}/{n_episodes}, score: {time}, e: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir + f"weights_{e:04d}.hdf5")


















