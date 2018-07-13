'''
Created on Oct 30, 2017

An DQN Agent modified for DDQ Agent

Some methods are not consistent with super class Agent.

@author: Baolin Peng
'''

import random, copy, json
import cPickle as pickle
import numpy as np
from collections import namedtuple, deque

from deep_dialog import dialog_config

from agent import Agent
from deep_dialog.qlearning import DQN

import torch
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device('cpu')

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))


class AgentDQN(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']

        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 5000)
        self.experience_replay_pool = deque(
            maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.experience_replay_pool_from_model = deque(
            maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.running_expereince_pool = None # hold experience from both user and world model

        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 5
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions).to(DEVICE)
        self.target_dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions).to(DEVICE)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()

        self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=1e-3)

        self.cur_bellman_err = 0

        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.load(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

    def state_to_action(self, state):
        """ DQN: Input state, output action """
        # self.state['turn'] += 2
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        if self.warm_start == 1:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        else:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action[0]])

        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        # turn_rep = np.zeros((1,1)) + state['turn'] / 10.
        turn_rep = np.zeros((1, 1))

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        # ########################################################################
        # #   Representation of KB results (scaled counts)
        # ########################################################################
        # kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
        #
        # ########################################################################
        # #   Representation of KB results (binary)
        # ########################################################################
        # kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

        self.final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation

    def run_policy(self, representation):
        """ epsilon-greedy policy """

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                return self.DQN_policy(representation)

    def rule_policy(self):
        """ Rule Policy """

        act_slot_response = {}

        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)

    def DQN_policy(self, state_representation):
        """ Return action from DQN"""

        with torch.no_grad():
            action = self.dqn.predict(torch.FloatTensor(state_representation))
        return action

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, st_user, from_model=False):
        """ Register feedback from either environment or world model, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        st_user = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over, st_user)

        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else:  # Prediction Mode
            if not from_model:
                self.experience_replay_pool.append(training_example)
            else:
                self.experience_replay_pool_from_model.append(training_example)

    def sample_from_buffer(self, batch_size):
        """Sample batch size examples from experience buffer and convert it to torch readable format"""
        # type: (int, ) -> Transition

        batch = [random.choice(self.running_expereince_pool) for i in xrange(batch_size)]
        np_batch = []
        for x in range(len(Transition._fields)):
            v = []
            for i in xrange(batch_size):
                v.append(batch[i][x])
            np_batch.append(np.vstack(v))

        return Transition(*np_batch)

    def train(self, batch_size=1, num_batches=100):
        """ Train DQN with experience buffer that comes from both user and world model interaction."""

        self.cur_bellman_err = 0.
        self.cur_bellman_err_planning = 0.
        self.running_expereince_pool = list(self.experience_replay_pool) + list(self.experience_replay_pool_from_model)

        for iter_batch in range(num_batches):
            for iter in range(len(self.running_expereince_pool) / (batch_size)):
                self.optimizer.zero_grad()
                batch = self.sample_from_buffer(batch_size)

                state_value = self.dqn(torch.FloatTensor(batch.state)).gather(1, torch.tensor(batch.action))
                next_state_value, _ = self.target_dqn(torch.FloatTensor(batch.next_state)).max(1)
                next_state_value = next_state_value.unsqueeze(1)
                term = np.asarray(batch.term, dtype=np.float32)
                expected_value = torch.FloatTensor(batch.reward) + self.gamma * next_state_value * (
                    1 - torch.FloatTensor(term))

                loss = F.mse_loss(state_value, expected_value)
                loss.backward()
                self.optimizer.step()
                self.cur_bellman_err += loss.item()

            if len(self.experience_replay_pool) != 0:
                print (
                    "cur bellman err %.4f, experience replay pool %s, model replay pool %s, cur bellman err for planning %.4f" % (
                        float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),
                        len(self.experience_replay_pool), len(self.experience_replay_pool_from_model),
                        self.cur_bellman_err_planning))

    # def train_one_iter(self, batch_size=1, num_batches=100, planning=False):
    #     """ Train DQN with experience replay """
    #     self.cur_bellman_err = 0
    #     self.cur_bellman_err_planning = 0
    #     running_expereince_pool = self.experience_replay_pool + self.experience_replay_pool_from_model
    #     for iter_batch in range(num_batches):
    #         batch = [random.choice(self.experience_replay_pool) for i in xrange(batch_size)]
    #         np_batch = []
    #         for x in range(5):
    #             v = []
    #             for i in xrange(len(batch)):
    #                 v.append(batch[i][x])
    #             np_batch.append(np.vstack(v))
    #
    #         batch_struct = self.dqn.singleBatch(np_batch)
    #         self.cur_bellman_err += batch_struct['cost']['total_cost']
    #         if planning:
    #             plan_step = 3
    #             for _ in xrange(plan_step):
    #                 batch_planning = [random.choice(self.experience_replay_pool) for i in
    #                                   xrange(batch_size)]
    #                 np_batch_planning = []
    #                 for x in range(5):
    #                     v = []
    #                     for i in xrange(len(batch_planning)):
    #                         v.append(batch_planning[i][x])
    #                     np_batch_planning.append(np.vstack(v))
    #
    #                 s_tp1, r, t = self.user_planning.predict(np_batch_planning[0], np_batch_planning[1])
    #                 s_tp1[np.where(s_tp1 >= 0.5)] = 1
    #                 s_tp1[np.where(s_tp1 <= 0.5)] = 0
    #
    #                 t[np.where(t >= 0.5)] = 1
    #
    #                 np_batch_planning[2] = r
    #                 np_batch_planning[3] = s_tp1
    #                 np_batch_planning[4] = t
    #
    #                 batch_struct = self.dqn.singleBatch(np_batch_planning)
    #                 self.cur_bellman_err_planning += batch_struct['cost']['total_cost']
    #
    #     if len(self.experience_replay_pool) != 0:
    #         print ("cur bellman err %.4f, experience replay pool %s, cur bellman err for planning %.4f" % (
    #             float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),
    #             len(self.experience_replay_pool), self.cur_bellman_err_planning))

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print 'saved model in %s' % (path,)
        except Exception, e:
            print 'Error: Writing model fails: %s' % (path,)
            print e

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']
        print "Trained DQN Parameters:", json.dumps(trained_file['params'], indent=2)
        return model

    def set_user_planning(self, user_planning):
        self.user_planning = user_planning

    def save(self, filename):
        torch.save(self.dqn.state_dict(), filename)

    def load(self, filename):
        self.dqn.load_state_dict(torch.load(filename))

    def reset_dqn_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
