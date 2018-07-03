from .usersim import UserSimulator
import argparse, json, random, copy, sys
import numpy as np
from user_model import SimulatorModel
from collections import namedtuple, deque
from deep_dialog import dialog_config

import torch
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'agent_action', 'next_state', 'reward', 'term', 'user_action'))


class ModelBasedSimulator(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """

        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.feasible_actions_users = dialog_config.feasible_actions_users
        self.num_actions = len(self.feasible_actions)
        self.num_actions_user = len(self.feasible_actions_users)

        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 9 * self.slot_cardinality + 3 + self.max_turn

        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']

        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 5000)

        self.learning_phase = params['learning_phase']
        self.hidden_size = params['hidden_size']

        self.training_examples = deque(maxlen=self.experience_replay_pool_size)

        self.predict_model = True

        self.model = SimulatorModel(self.num_actions, self.hidden_size, self.state_dimension, self.num_actions_user, 1)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)

    def initialize_episode(self):
        """ Initialize a new episode (dialog)
        """

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        self.goal = self._sample_goal(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        # sample first action
        user_action = self._sample_action()
        assert (self.episode_over != 1), ' but we just started'
        return user_action

    def _sample_action(self):
        """ randomly sample a start action based on user goal """

        self.state['diaact'] = random.choice(dialog_config.start_dia_acts.keys())

        # "sample" informed slots
        if len(self.goal['inform_slots']) > 0:
            known_slot = random.choice(self.goal['inform_slots'].keys())
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

            if 'moviename' in self.goal['inform_slots'].keys():  # 'moviename' must appear in the first user turn
                self.state['inform_slots']['moviename'] = self.goal['inform_slots']['moviename']

            for slot in self.goal['inform_slots'].keys():
                if known_slot == slot or slot == 'moviename': continue
                self.state['rest_slots'].append(slot)

        self.state['rest_slots'].extend(self.goal['request_slots'].keys())

        # "sample" a requested slot
        request_slot_set = list(self.goal['request_slots'].keys())
        request_slot_set.remove('ticket')
        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
        else:
            request_slot = 'ticket'
        self.state['request_slots'][request_slot] = 'UNK'

        if len(self.state['request_slots']) == 0:
            self.state['diaact'] = 'inform'

        if (self.state['diaact'] in ['thanks', 'closing']):
            self.episode_over = True  # episode_over = True
        else:
            self.episode_over = False  # episode_over = False

        sample_action = {}
        sample_action['diaact'] = self.state['diaact']
        sample_action['inform_slots'] = self.state['inform_slots']
        sample_action['request_slots'] = self.state['request_slots']
        sample_action['turn'] = self.state['turn']

        self.add_nl_to_action(sample_action)
        return sample_action

    def _sample_goal(self, goal_set):
        """ sample a user goal  """

        self.sample_goal = random.choice(self.start_set[self.learning_phase])
        return self.sample_goal

    def prepare_user_goal_representation(self, user_goal):
        """"""

        request_slots_rep = np.zeros((1, self.slot_cardinality))
        inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for s in user_goal['request_slots']:
            s = s.strip()
            request_slots_rep[0, self.slot_set[s]] = 1
        for s in user_goal['inform_slots']:
            s = s.strip()
            inform_slots_rep[0, self.slot_set[s]] = 1
        self.user_goal_representation = np.hstack([request_slots_rep, inform_slots_rep])

        return self.user_goal_representation

    def sample_from_buffer(self, batch_size):
        """Sample batch size examples from experience buffer and convert it to torch readable format"""

        batch = [random.choice(self.training_examples) for i in xrange(batch_size)]
        np_batch = []

        for x in range(len(Transition._fields)):
            v = []
            for i in xrange(batch_size):
                v.append(batch[i][x])
            np_batch.append(np.vstack(v))

        return Transition(*np_batch)

    def train(self, batch_size=1, num_batches=1):
        """
        Train the world model with all the accumulated examples
        :param batch_size: self-explained
        :param num_batches: self-explained
        :return: None
        """
        self.total_loss = 0
        for iter_batch in range(num_batches):
            for iter in range(len(self.training_examples) / (batch_size)):
                self.optimizer.zero_grad()

                batch = self.sample_from_buffer(batch_size)
                state = torch.FloatTensor(batch.state)
                action = torch.LongTensor(batch.agent_action)
                reward = torch.FloatTensor(batch.reward)
                term = torch.FloatTensor(np.asarray(batch.term, dtype=np.int32))
                user_action = torch.LongTensor(batch.user_action).squeeze(1)

                reward_, term_, user_action_ = self.model(state, action)

                loss = F.mse_loss(reward_, reward) + \
                       F.binary_cross_entropy_with_logits(term_, term) + \
                       F.nll_loss(user_action_, user_action)
                loss.backward()

                self.optimizer.step()
                self.total_loss += loss.item()

            print ("Total cost for user modeling: %.4f, training replay pool %s" % (
                float(self.total_loss) / (float(len(self.training_examples)) / float(batch_size)),
                len(self.training_examples)))

    def train_by_iter(self, batch_size=1, num_batches=1):
        """
        Train the model with num_batches examples.
        :param batch_size:
        :param num_batches:
        :return: None
        """
        self.total_loss = 0
        for iter_batch in range(num_batches):
            self.optimizer.zero_grad()
            batch = self.sample_from_buffer(batch_size)
            state = torch.FloatTensor(batch.state)
            action = torch.LongTensor(batch.agent_action)
            reward = torch.FloatTensor(batch.reward)
            term = torch.FloatTensor(np.asarray(batch.term, dtype=np.int32))
            user_action = torch.LongTensor(batch.user_action).squeeze(1)

            reward_, term_, user_action_ = self.model(state, action)

            loss = F.mse_loss(reward_, reward) + \
                   F.binary_cross_entropy_with_logits(term_, term) + \
                   F.nll_loss(user_action_, user_action)
            loss.backward()

            self.optimizer.step()
            self.total_loss = loss.item()

            print ("Total cost for user modeling: %.4f, training replay pool %s" % (
                float(self.total_loss), len(self.training_examples)))

    def next(self, s, a):
        """
        Provide
        :param s: state representation from tracker
        :param a: last action from agent
        :return: next user action, termination and reward predicted by world model
        """

        self.state['turn'] += 2
        if (self.max_turn > 0 and self.state['turn'] >= self.max_turn):
            reward = - self.max_turn
            term = True
            self.state['request_slots'].clear()
            self.state['inform_slots'].clear()
            self.state['diaact'] = "closing"
            response_action = {}
            response_action['diaact'] = self.state['diaact']
            response_action['inform_slots'] = self.state['inform_slots']
            response_action['request_slots'] = self.state['request_slots']
            response_action['turn'] = self.state['turn']
            return response_action, term, reward

        s = self.prepare_state_representation(s)
        g = self.prepare_user_goal_representation(self.sample_goal)
        s = np.hstack([s, g])
        reward, term, action = self.predict(torch.FloatTensor(s), torch.LongTensor(np.asarray(a)[:, None]))
        action = action.item()
        reward = reward.item()
        term = term.item()
        action = copy.deepcopy(self.feasible_actions_users[action])

        if action['diaact'] == 'inform':
            if len(action['inform_slots'].keys()) > 0:
                slots = action['inform_slots'].keys()[0]
                if slots in self.sample_goal['inform_slots'].keys():
                    action['inform_slots'][slots] = self.sample_goal['inform_slots'][slots]
                else:
                    action['inform_slots'][slots] = dialog_config.I_DO_NOT_CARE

        response_action = action

        term = term > 0.5

        if reward > 1:
            reward = 2 * self.max_turn
        elif reward < -1:
            reward = -self.max_turn
        else:
            reward = -1

        return response_action, term, reward

    def predict(self, s, a):
        return self.model.predict(s, a)

    def register_user_goal(self, goal):
        self.user_goal = goal

    def action_index(self, act_slot_response):
        """ Return the index of action """
        del act_slot_response['turn']
        del act_slot_response['nl']

        for i in act_slot_response['inform_slots'].keys():
            act_slot_response['inform_slots'][i] = 'PLACEHOLDER'

        for (i, action) in enumerate(self.feasible_actions_users):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None

    def register_experience_replay_tuple(self, s_t, agent_a_t, s_tplus1, reward, term, user_a_t):
        """ Register feedback from the environment, to be stored as future training data for world model"""

        state_t_rep = self.prepare_state_representation(s_t)
        goal_rep = self.prepare_user_goal_representation(self.sample_goal)
        state_t_rep = np.hstack([state_t_rep, goal_rep])
        agent_action_t = agent_a_t
        user_action_t = user_a_t
        action_idx = self.action_index(copy.deepcopy(user_a_t))
        reward_t = reward
        term_t = term

        if reward_t > 1:
            reward_t = 1
        elif reward_t < -1:
            reward_t = -1
        elif reward_t == -1:
            reward_t = -0.1

        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example_for_user = (state_t_rep, agent_action_t, state_tplus1_rep, reward_t, term, action_idx)

        if self.predict_model:
            self.training_examples.append(training_example_for_user)


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

        # turn_rep = np.zeros((1, 1)) + state['turn'] / 10.
        turn_rep = np.zeros((1, 1))

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))
        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

        # kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
        #
        # ########################################################################
        # #   Representation of KB results (binary)
        # ########################################################################
        # kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
        #     kb_results_dict['matching_all_constraints'] > 0.)
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation
