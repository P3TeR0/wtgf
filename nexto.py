import numpy as np
from game_state import GameState
from nexto_obs import NextoObsBuilder
from controller_state import SimpleControllerState
from agent import Agent


class Nexto:
    def __init__(self, buffer_cursor, total_boosts, total_players):
        self.team = 0
        self.index = 0
        self.obs_builder = None
        self.buffer_cursor = buffer_cursor
        self.total_boosts = total_boosts
        self.total_players = total_players
        self.game_state: GameState = GameState(total_boosts)
        self.controls = None
        self.action = None
        self.agent = Agent()

    def initialize_agent(self):
        temp_gamestate = GameState(self.total_boosts)
        temp_gamestate.decode(self.buffer_cursor)
        self.obs_builder = NextoObsBuilder(self.total_players, self.total_boosts, self.buffer_cursor)
        self.game_state = GameState(self.total_boosts)
        self.index = temp_gamestate.index
        self.team = temp_gamestate.players[self.index].team_num
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)

    def get_output(self) -> SimpleControllerState:
        self.game_state = GameState(self.total_boosts)
        self.game_state.decode(self.buffer_cursor)

        player = self.game_state.players[self.index]
        teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
        opponents = [p for p in self.game_state.players if p.team_num != self.team]

        self.game_state.players = [player] + teammates + opponents

        obs = self.obs_builder.build_obs(player, self.game_state, self.action)

        self.action, weights = self.agent.act(obs, 1)
        self.update_controls(self.action)

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
