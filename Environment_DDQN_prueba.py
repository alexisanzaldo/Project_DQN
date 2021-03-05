from gym import Env
import gym
from utils import plotLearning
from utils import plot_learning_curve
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import random
import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import matplotlib.pyplot as plt


def action_map(action_index, power_avail):
    selected_action = power_avail[action_index]
    return selected_action


def BS_UE_distances(x_sample, y_sample, x_user, y_user):
    distances = np.zeros((len(x_sample), len(x_user)))

    for i in range(len(x_sample)):
        for j in range(len(x_user)):
            distances[i, j] = np.sqrt(((x_user[j] - x_sample[i]) ** 2) + ((y_user[j] - y_sample[i]) ** 2))
    return distances


def association(distances, UEs, BSs):
    counter = np.arange(UEs)
    UE_BS_index = np.ndarray.argmin(distances, 0)
    BS_load = np.zeros(BSs).astype(int)

    while np.count_nonzero(BS_load) != BSs:
        for i in np.arange(BSs):
            if not (len(list(filter(lambda x: x == i, UE_BS_index))) > 0):  # check if item i is on array UE_BS_index
                UE_BS_index[np.ndarray.argmin(distances[i, counter], 0)] = i
                counter = np.delete(counter, np.ndarray.argmin(distances[i, counter], 0))

        BS_load = [np.count_nonzero(UE_BS_index == i) for i in np.arange(BSs)]

    return UE_BS_index, BS_load


def dual_slope_path_loss_matrix(d, K0, alfa1, alfa2, dBP, path_loss):
    path_loss[(d <= dBP)] = K0 + 10 * alfa1 * np.log10(d[(d <= dBP)])
    path_loss[(d > dBP)] = K0 + 10 * alfa2 * np.log10(d[(d > dBP)]) - 10 * (alfa2 - alfa1) * np.log10(dBP)
    return path_loss


def path_loss_LTE(d):
    # d in kilometers d = d/1000
    path_loss = 120.9 + 37.6 * np.log10(d / 1000)
    return path_loss


def shadowing_fading(base_stations, users, dv=8):
    shadowing = dv * np.random.normal(size=(base_stations, users))
    return shadowing


def scheduling_per_access_point(UE_BS_index, scheduling_counter, channel_gain, active_users, BSs):
    for i in np.arange(BSs):
        index = get_indexes(i, UE_BS_index)
        if len((np.where(scheduling_counter[index] == 0))[0]) != 0:  # hay mas de un minimo
            aux = (np.where(scheduling_counter[index] == 0)[0])  # Indice de todos los minimos

            aux = aux.astype(int)
            index = np.array(index)
            index = index.astype(int)

            idx = (np.where(channel_gain[i, index[aux]] == channel_gain[i, index[aux]].max()))[
                0]  # indice de ganancia maxima
            active_users[i] = index[aux[idx]]  # indice del elegido
            scheduling_counter[index[aux[idx]]] = 1
        else:
            scheduling_counter[index] = scheduling_counter[index] * 0
            aux = (np.where(scheduling_counter[index] == 0)[0])  # Indice de todos los minimos

            aux = aux.astype(int)
            index = np.array(index)
            index = index.astype(int)

            idx = (np.where(channel_gain[i, index[aux]] == channel_gain[i, index[aux]].max()))[
                0]  # indice de ganancia maxima
            active_users[i] = index[aux[idx]]  # indice del elegido
            scheduling_counter[index[aux[idx]]] = 1
    return active_users, scheduling_counter


def get_indexes(x, xs):
    indexes = [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    return indexes


def grid_deployment(Nbs, Rcell):
    cell_position = np.zeros((Nbs, 2))
    if (Nbs > 1):
        theta = np.arange(0, Nbs - 1) * np.pi / 3  # en matlab el vector es vertical
        cell_position[1:, :] = np.sqrt(3) * Rcell * np.concatenate(([np.cos(theta)], [np.sin(theta)]), axis=0).T

    if (Nbs > 7):
        theta = np.arange(start=-np.pi / 6, stop=(5 / 3) * np.pi, step=np.pi / 3)
        x = 3 * Rcell * np.cos(theta)
        y = 3 * Rcell * np.sin(theta)
        theta = np.arange(start=0, stop=(5 / 3) * np.pi, step=np.pi / 3)
        x = np.reshape(np.concatenate(([x], [2 * np.sqrt(3) * Rcell * np.cos(theta)]), axis=0), (2 * theta.size, 1),
                       order='F')
        y = np.reshape(np.concatenate(([y], [2 * np.sqrt(3) * Rcell * np.sin(theta)]), axis=0), (2 * theta.size, 1),
                       order='F')

        if Nbs > 19:
            cell_position[7:19, 0:2] = np.concatenate((x, y), axis=1)
        else:
            cell_position[7:Nbs + 1, 0:2] = np.concatenate((x[0:Nbs - 7], y[0:Nbs - 7]), axis=1)

    if Nbs > 19 and Nbs < 38:
        theta = np.arange(start=-np.arcsin(3 / np.sqrt(21)), stop=(5 / 3) * np.pi, step=np.pi / 3)
        x1 = np.sqrt(21) * Rcell * np.cos(theta)
        y1 = np.sqrt(21) * Rcell * np.sin(theta)
        theta = np.arange(start=-np.arcsin(3 / 2 / np.sqrt(21)), stop=(5 / 3) * np.pi, step=np.pi / 3)
        x2 = np.sqrt(21) * Rcell * np.cos(theta)
        y2 = np.sqrt(21) * Rcell * np.sin(theta)
        theta = np.arange(start=0, stop=(5 / 3) * np.pi, step=np.pi / 3)
        x3 = 3 * np.sqrt(3) * Rcell * np.cos(theta)
        y3 = 3 * np.sqrt(3) * Rcell * np.sin(theta)
        x = np.reshape(np.concatenate(([x1], [x2], [x3]), axis=0), (x1.size + x2.size + x3.size, 1), order='F')
        y = np.reshape(np.concatenate(([y1], [y2], [y3]), axis=0), (y1.size + y2.size + y3.size, 1), order='F')
        cell_position[19:Nbs + 1, 0:2] = np.concatenate((x[0:Nbs - 19], y[0:Nbs - 19]), axis=1)

    x_base_station = cell_position[:, 0]
    y_base_station = cell_position[:, 1]

    return x_base_station, y_base_station, cell_position






class CellularEnv(Env):
    def __init__(self, x_base_station, y_base_station, Nue, Nbs, Rmin, Rmax, cell_position, path_loss, A, intervals,
                 Pmin_dBm, Pmax_dBm, noise_power_dBm, SINR_th, SINR_cap):
        self.x_base_station = x_base_station
        self.y_base_station = y_base_station
        self.Nue = Nue
        self.Nbs = Nbs
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Pmin_dBm = Pmin_dBm
        self.Pmax_dBm = Pmax_dBm
        self.Pmax = (10 ** ((self.Pmax_dBm - 30) / 10))
        self.Pmin = (10 ** ((self.Pmin_dBm - 30) / 10))
        self.noise_power_dBm = noise_power_dBm
        self.intervals = intervals
        self.path_loss = path_loss
        self.cell_position = cell_position
        self.scheduling_counter = np.zeros(self.Nue)
        self.active_users = np.zeros(self.Nbs).astype(int)
        self.SINR_th = np.array([SINR_th])
        self.SINR_cap = np.array(SINR_cap)
        self.A = A
        self.rates = []
        self.power = []
        self.SINR = []
        self.episode_length = 0
        #self.reward_rate = np.zeros(self.Nbs)
        self.reward_rate = 0

        # Is this ok?
        self.P_ = np.zeros(self.Nbs)
        self.R_ = np.zeros(self.Nbs)
        self.SINR_ = np.zeros(self.Nbs)
        # self.power_available = np.arange(A+1)*self.Pmax/A
        #self.power_available = np.linspace(self.Pmin, self.Pmax, A)

        #self.power_available = (10 ** ((np.linspace(self.Pmin_dBm, self.Pmax_dBm, A) - 30) / 10))
        self.power_available = np.hstack(
            [np.zeros((1), dtype=np.float32), 1e-3 * pow(10., np.linspace(self.Pmin_dBm, self.Pmax_dBm, self.A - 1) / 10.)])

        self.N = (10 ** ((self.noise_power_dBm - 30) / 10))
        self.Tx_power = np.zeros(self.Nbs)

        self.scheme_full_reuse = 1

        # Communication Channel Components ----------------------------------------
        self.K0 = 39
        self.alfa1 = 2
        self.alfa2 = 4
        self.dBP = 100

        self.state = np.zeros((self.Nbs, self.Nbs * 4), dtype=np.float32)  # {G, P_, R_}

        # self.observation_space = Box(low=np.array([0]), high=np.array([100])) # Este espacio de observacion no es adecuado
        high = np.ones(self.Nue * 3) * 100
        low = np.zeros(self.Nue * 3)
        self.observation_space = Box(low=low, high=high)
        self.action_space = Discrete(self.A)

        # Set start temp **** (?start channel?)

        # 0. Despliegue de usuarios
        user_position = np.zeros((self.Nue, 2)) # User deployment, one UE per BS *
        R = np.random.uniform(low=self.Rmin, high=self.Rmax, size=(1, self.Nue))
        angle = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(1, self.Nue))
        user_position[:, 0] = R * np.cos(angle)
        user_position[:, 1] = R * np.sin(angle)
        user_position = self.cell_position + user_position
        x_user = user_position[:, 0]
        y_user = user_position[:, 1]
        # 1. Calcular la distancia entre estaciones base y usuarios
        distances = BS_UE_distances(self.x_base_station, self.y_base_station, x_user, y_user)
        # 2. Asociar por la distancia mínima y Calcular la carga de cada estacion base
        UE_BS_index, BS_load = association(distances, self.Nbs, self.Nue)
        self.UE_BS_index = UE_BS_index  # Este valor no va a cambiar nunca
        # Initialization
        #scheduling_counter = np.zeros(len(self.UE_BS_index))
        # 3. Calcular el path loss para el canal de transmision y los canales interferentes
        self.path_loss = dual_slope_path_loss_matrix(distances, self.K0, self.alfa1, self.alfa2, self.dBP,
                                                     self.path_loss)
        # 4.1. Calcular el desvanecimiento por sombreo del canal
        shadowing = shadowing_fading(self.Nbs, self.Nue, dv=8)

        variance = 10 ** (-(self.path_loss + shadowing) / 10)
        channel = np.sqrt(variance / 2) * (
                np.random.normal(size=(self.Nbs, self.Nue)) + 1j * np.random.normal(size=(self.Nbs, self.Nue)))
        self.channel_gain = (abs(channel) ** 2)

        if self.scheme_full_reuse == 1:
            self.active_users, self.scheduling_counter = scheduling_per_access_point(
                self.UE_BS_index, self.scheduling_counter, self.channel_gain, self.active_users, self.Nbs)
        # ------------------------------------------------------------------------------------------------------------

        # Prueba de normalizacion del canal
        #channel_gain_aux = (self.channel_gain - self.channel_gain.min()) / \
        #                   (self.channel_gain.max() - self.channel_gain.min())


        # State formating --------------------------------------------------------------------------------------------
        for i_agent_BS in np.arange(self.Nbs):
            idy = self.active_users[i_agent_BS]
            idx = self.UE_BS_index[idy]
            #channel_state = np.roll(channel_gain_aux[self.active_users, idy], -idx)
            # ORIGINAL vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            channel_state = np.roll(self.channel_gain[self.active_users, idy], -idx)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            power_state = np.roll(self.P_, -i_agent_BS)  # How to define?
            reward_state = np.roll(self.R_, -i_agent_BS)  # How to define?
            sinr_state = np.roll(self.SINR_, -i_agent_BS)

            self.state[i_agent_BS, :] = np.hstack((channel_state, power_state, reward_state, sinr_state))
        # State formating --------------------------------------------------------------------------------------------

    def step(self, action):
        #old_state = self.state
        old_state = np.zeros((self.Nbs, self.Nbs * 4), dtype=np.float32)  # {G, P_, R_}
        # Prueba de normalizacion del canal
        #channel_gain_aux = (self.channel_gain - self.channel_gain.min()) / \
        #                   (self.channel_gain.max() - self.channel_gain.min())
        # State formating --------------------------------------------------------------------------------------------
        for i_agent_BS in np.arange(self.Nbs):
            idy = self.active_users[i_agent_BS]
            idx = self.UE_BS_index[idy]
            #channel_state = np.roll(channel_gain_aux[self.active_users, idy], -idx)
            # ORIGINAL vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            channel_state = np.roll(self.channel_gain[self.active_users, idy], -idx)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            power_state = np.roll(self.P_, -i_agent_BS)  # How to define?
            reward_state = np.roll(self.R_, -i_agent_BS)  # How to define?
            sinr_state = np.roll(self.SINR_, -i_agent_BS)

            old_state[i_agent_BS, :] = np.hstack((channel_state, power_state, reward_state, sinr_state))

        # ------------------------------Noise------------------------------------------------------------------------
        # Shadowing - Every step is considered a time intervals, in wich the shadowind and Rayleigh fading is performed
        shadowing = shadowing_fading(self.Nbs, self.Nue, dv=8)
        variance = 10 ** (-(self.path_loss + shadowing) / 10)
        channel = np.sqrt(variance / 2) * (
                np.random.normal(size=(self.Nbs, self.Nue)) + 1j * np.random.normal(size=(self.Nbs, self.Nue)))
        self.channel_gain = (abs(channel) ** 2)
        # ------------------------------Noise------------------------------------------------------------------------

        # --------------------- Reward Calculation -------------------------------------------------------------------
        for i in np.arange(self.Nbs): # Action is a vector of len(Nbs)
            self.Tx_power[i] = action_map(action[i], self.power_available)

        for i in np.arange(self.Nbs):
            if self.active_users[i] is None:
                S = 0
                C = 0
                Interference = 0
                self.rates.append(C)
                self.SINR.append(0)
            else:
                S = self.channel_gain[i, self.active_users[i]] * self.Tx_power[i]
                # Tomando en cuenta que las estaciones base son los renglones (primer indice) y los usuarios las columnas ... -> shape (Nbs, Nue)
                aux = [x for j, x in enumerate(self.active_users) if (j != i) and (x != None)]
                Interference = np.sum(self.channel_gain[self.UE_BS_index[aux], i] * self.Tx_power[aux])

                sinr = np.minimum(S / (Interference + self.N), self.SINR_cap)  # Maximum SINR Restriction


                C = math.log2(1 + sinr)
                '''
                if sinr >= self.SINR_th:  # Minimum SINR Restriction
                    C = math.log2(1 + sinr)
                else:
                    C = 0
                    sinr=0
                '''

                self.rates.append(C)  # Todas las tasas concatenadas
                self.SINR.append(sinr)
                #self.rates_per_UE[self.active_users[
                #                      i], self.episode_length] = C  # R_j_i (t) : Tasa instantanea en en el intervalo t. Matriz (24UEs x SchedulingIntervals)

        sumrate = np.mean(self.rates[(self.episode_length * self.Nbs):(self.episode_length * self.Nbs) + self.Nbs])
        #self.reward_rate =self.rates[(self.episode_length * self.Nbs):(self.episode_length * self.Nbs) + self.Nbs]
        #self.reward_rate += sumrate
        self.reward_rate = sumrate


        # State formating -------------------------------------------------------------------------------------------
        self.power = np.hstack((self.power, self.Tx_power))
        # El primer intervalo P_ y R_ deben de ser igual a 0
        self.P_ = self.power[(self.episode_length * self.Nbs):(self.episode_length * self.Nbs) + self.Nbs]
        # Power Normalization ---------------------------------------------------------------------------------------
        # Pmin deberia ser igual a 0
        #self.P_ = (self.P_ - self.Pmin) / (self.Pmax - self.Pmin)

        #self.P_[self.P_ <= 0] = 0
        # Power Normalization ---------------------------------------------------------------------------------------
        self.R_ = self.rates[(self.episode_length * self.Nbs):(self.episode_length * self.Nbs) + self.Nbs]
        self.SINR_ = self.SINR[(self.episode_length * self.Nbs):(self.episode_length * self.Nbs) + self.Nbs]
        # SINR Normalization ----------------------------------------------------------------------------------------
        #self.SINR_ = (self.SINR_ - self.SINR_th) / (self.SINR_cap-self.SINR_th)
        #self.SINR_[self.SINR_ <= 0] = 0
        # SINR Normalization ----------------------------------------------------------------------------------------
        # Normalizacion de la ganancia ------------------------------------------------------------------------------
        #channel_gain_aux = (self.channel_gain - self.channel_gain.min()) / \
        #                   (self.channel_gain.max() - self.channel_gain.min())


        # State formating -------------------------------------------------------------------------------------------
        for i_agent_BS in np.arange(self.Nbs):
            idy = self.active_users[i_agent_BS]
            idx = self.UE_BS_index[idy]
            #channel_state = np.roll(channel_gain_aux[self.active_users, idy], -idx)
            # ORIGINAL vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            channel_state = np.roll(self.channel_gain[self.active_users, idy], -idx)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            power_state = np.roll(self.P_, -i_agent_BS)  # / self.Pmax # Normalization? # How to define?
            reward_state = np.roll(self.R_, -i_agent_BS)  # How to define?
            sinr_state = np.roll(self.SINR_, -i_agent_BS)
            self.state[i_agent_BS, :] = np.hstack((channel_state, power_state, reward_state, sinr_state))
            #print('Recompensa', reward_state)
            #print('power_state', power_state)
            #wait = input('Press enter to continue')


        # Set placeholder for info # Requierement of OPENAI Environments
        info = {}

        self.episode_length += 1
        # Check if episode is done
        if self.episode_length >= self.intervals:
            done = True
        else:
            done = False

        #print('Estado al principio del estado (within env)', old_state[0])
        #print('Estado al final del intervalo (within env)', self.state[0])

        return old_state, self.state, self.reward_rate, done, info, sumrate

        #return state, self.reward_rate, done, info, sumrate

    def reset(self):
        # Reset UE deployment ----------------------------------------------------------------------------------------
        self.scheduling_counter = np.zeros(self.Nue)
        #self.active_users = np.zeros(self.Nbs).astype(int)
        # -- Se mantienen el estado anterior al reiniciar el episodio, ya que no se puede tomar una decisión con 0s---
        self.rates = []
        self.SINR = []
        self.power = []
        self.reward_rate = 0
        #self.reward_rate = np.zeros(self.Nbs)
        #self.rates_per_UE = np.zeros((self.Nue, self.intervals))

        user_position = np.zeros((self.Nue, 2))

        R = np.random.uniform(low=self.Rmin, high=self.Rmax, size=(1, self.Nue))
        angle = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(1, self.Nue))
        user_position[:, 0] = R * np.cos(angle)
        user_position[:, 1] = R * np.sin(angle)
        user_position = self.cell_position + user_position

        x_user = user_position[:, 0]
        y_user = user_position[:, 1]

        # 1. Calcular la distancia entre estaciones base y usuarios
        distances = BS_UE_distances(self.x_base_station, self.y_base_station, x_user, y_user)
        # 2. Asociar por la distancia mínima y Calcular la carga de cada estacion base
        UE_BS_index, BS_load = association(distances, self.Nbs, self.Nue)
        self.UE_BS_index = UE_BS_index
        # Initialization
        #scheduling_counter = np.zeros(len(self.UE_BS_index))
        # 3. Calcular el path loss para el canal de transmision y los canales interferentes
        self.path_loss = dual_slope_path_loss_matrix(distances, self.K0, self.alfa1, self.alfa2, self.dBP,
                                                     self.path_loss)

        self.episode_length = 0

        # State formating ----------------------------------------------------------
        for i_agent_BS in np.arange(self.Nbs):
            idy = self.active_users[i_agent_BS]
            idx = self.UE_BS_index[idy]
            channel_state = np.roll(self.channel_gain[self.active_users, idy], -idx)
            power_state = np.roll(self.P_, -i_agent_BS)  # / self.Pmax      # How to define?
            reward_state = np.roll(self.R_, -i_agent_BS)  # How to define?
            sinr_state = np.roll(self.SINR_, -i_agent_BS)

            self.state[i_agent_BS, :] = np.hstack((channel_state, power_state, reward_state, sinr_state))


        return self.state
