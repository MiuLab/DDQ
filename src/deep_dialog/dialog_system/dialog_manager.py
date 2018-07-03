"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

import json
from . import StateTracker
from deep_dialog import dialog_config
import copy


class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """

    def __init__(self, agent, user, world_model, act_set, slot_set, movie_dictionary):
        self.agent = agent
        self.user = user
        self.world_model = world_model
        self.act_set = act_set
        self.slot_set = slot_set
        self.state_tracker = StateTracker(act_set, slot_set, movie_dictionary)
        self.user_action = None
        self.reward = 0
        self.episode_over = False


        self.use_world_model = False
        self.running_user = self.user

    def initialize_episode(self, use_environment=False):
        """ Refresh state for new dialog """

        self.reward = 0
        self.episode_over = False

        self.state_tracker.initialize_episode()
        self.running_user = self.user
        self.use_world_model = False

        if not use_environment:
            self.running_user = self.world_model
            self.use_world_model = True
        else:
            self.running_user = self.user
            self.use_world_model = False

        self.user_action = self.running_user.initialize_episode()

        if use_environment:
            self.world_model.sample_goal = self.user.sample_goal

        self.state_tracker.update(user_action=self.user_action)

        if dialog_config.run_mode < 3:
            print ("New episode, user goal:")
            print json.dumps(self.user.goal, indent=2)
        self.print_function(user_action=self.user_action)

        self.agent.initialize_episode()

    def next_turn(self, record_training_data=True, record_training_data_for_user=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """

        ########################################################################
        #   CALL AGENT TO TAKE HER TURN
        ########################################################################
        self.state = self.state_tracker.get_state_for_agent()
        self.agent_action = self.agent.state_to_action(self.state)

        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        self.state_tracker.update(agent_action=self.agent_action)

        self.state_user = self.state_tracker.get_state_for_user()

        self.agent.add_nl_to_action(self.agent_action)  # add NL to Agent Dia_Act
        self.print_function(agent_action=self.agent_action['act_slot_response'])

        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################
        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        if self.use_world_model:
            self.user_action, self.episode_over, self.reward = self.running_user.next(self.state_user,
                                                                                      self.agent.action)
        else:
            self.user_action, self.episode_over, dialog_status = self.running_user.next(self.sys_action)
            self.reward = self.reward_function(dialog_status)

        ########################################################################
        #   Update state tracker with latest user action
        ########################################################################
        if self.episode_over != True:
            self.state_tracker.update(user_action=self.user_action)
            self.print_function(user_action=self.user_action)

        self.state_user_next = self.state_tracker.get_state_for_agent()

        ########################################################################
        #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over, s_t_u, user_world_model)
        ########################################################################
        if record_training_data:
            self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward,
                                                        self.state_tracker.get_state_for_agent(), self.episode_over,
                                                        self.state_user, self.use_world_model)

        ########################################################################
        #  Inform world model of the outcome for this timestep
        # (s_t, a_t, s_{t+1}, r, t, ua_t)
        ########################################################################

        if record_training_data_for_user and not self.use_world_model:
            self.world_model.register_experience_replay_tuple(self.state_user, self.agent.action,
                                                              self.state_user_next, self.reward, self.episode_over,
                                                              self.user_action)

        return (self.episode_over, self.reward)

    def reward_function(self, dialog_status):
        """ Reward Function 1: a reward function based on the dialog_status """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = -self.user.max_turn  # 10
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2 * self.user.max_turn  # 20
        else:
            reward = -1
        return reward

    def reward_function_without_penalty(self, dialog_status):
        """ Reward Function 2: a reward function without penalty on per turn and failure dialog """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = 0
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2 * self.user.max_turn
        else:
            reward = 0
        return reward

    def print_function(self, agent_action=None, user_action=None):
        """ Print Function """

        if agent_action:
            if dialog_config.run_mode == 0:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            elif dialog_config.run_mode == 1:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (
                        agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'],
                        agent_action['request_slots']))
            elif dialog_config.run_mode == 2:  # debug mode
                print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (
                    agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'],
                    agent_action['request_slots']))
                print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))

            if dialog_config.auto_suggest == 1:
                print(
                    '(Suggested Values: %s)' % (
                    self.state_tracker.get_suggest_slots_values(agent_action['request_slots'])))
        elif user_action:
            if dialog_config.run_mode == 0:
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            elif dialog_config.run_mode == 1:
                print ("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (
                    user_action['turn'], user_action['diaact'], user_action['inform_slots'],
                    user_action['request_slots']))
            elif dialog_config.run_mode == 2:  # debug mode, show both
                print ("Turn %d usr: %s, inform_slots: %s, request_slots: %s" % (
                    user_action['turn'], user_action['diaact'], user_action['inform_slots'],
                    user_action['request_slots']))
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))

            if self.agent.__class__.__name__ == 'AgentCmd':  # command line agent
                user_request_slots = user_action['request_slots']
                if 'ticket' in user_request_slots.keys(): del user_request_slots['ticket']
                if len(user_request_slots) > 0:
                    possible_values = self.state_tracker.get_suggest_slots_values(user_action['request_slots'])
                    for slot in possible_values.keys():
                        if len(possible_values[slot]) > 0:
                            print('(Suggested Values: %s: %s)' % (slot, possible_values[slot]))
                        elif len(possible_values[slot]) == 0:
                            print('(Suggested Values: there is no available %s)' % (slot))
                else:
                    kb_results = self.state_tracker.get_current_kb_results()
                    print ('(Number of movies in KB satisfying current constraints: %s)' % len(kb_results))
