"""
Random Agent
"""

import sys
import argparse
import logging
import asyncio
import websockets
import json
from collections import defaultdict
import random

import numpy as np


logger = logging.getLogger(__name__)
games = {}
agentclass = None


class DotsAndBoxesAgent:
    """Example Dots and Boxes agent implementation base class.
    It returns a random next move.

    A DotsAndBoxesAgent object should implement the following methods:
    - __init__
    - add_player
    - register_action
    - next_action
    - end_game

    This class does not necessarily use the best data structures for the
    approach you want to use.
    """

    def __init__(self, player):
        """Create Dots and Boxes agent.

        :param player: Player number, 1 or 2
        """
        self.player = {player}
        self.ended = False
        self.board = np.zeros((3, 3))

    def add_player(self, player):
        """Use the same agent for multiple players."""
        self.player.add(player)

    def register_action(self, row, column, player):
        """Register action played in game.

        :param row:
        :param columns:
        :param player: 1 or 2
        """
        self.board[row][column] = {1: 1, 2: -1}[player]

    def next_action(self):
        """Return the next action this agent wants to perform.

        In this example, the function implements a random move. Replace this
        function with your own approach.

        :return: (row, column)
        """
        # Random move
        free_cells = np.transpose(np.where(self.board == 0))
        if len(free_cells) == 0:
            return None
        move = np.random.choice(len(free_cells))
        r, c = free_cells[move]
        return int(r), int(c)

    def end_game(self):
        self.ended = True


# MAIN EVENT LOOP

async def handler(websocket, path):
    logger.info("Start listening")
    game = None
    # msg = await websocket.recv()
    try:
        async for msg in websocket:
            logger.info("< {}".format(msg))
            try:
                msg = json.loads(msg)
            except json.decoder.JSONDecodeError as err:
                logger.error(err)
                return False
            game = msg["game"]
            answer = None
            if msg["type"] == "start":
                if msg["game"] in games:
                    games[msg["game"]].add_player(msg["player"])
                else:
                    games[msg["game"]] = agentclass(msg["player"])
                if msg["player"] == 1:
                    nm = games[game].next_action()
                    if nm is None:
                        logger.info("Game over")
                        continue
                    r, c = nm
                    answer = {
                        'type': 'action',
                        'location': [r, c],
                    }
                else:
                    answer = None

            elif msg["type"] == "action":
                r, c = msg["location"]
                games[game].register_action(r, c, msg["player"])
                if msg["nextplayer"] in games[game].player:
                    nm = games[game].next_action()
                    if nm is None:
                        logger.info("Game over")
                        continue
                    nr, nc = nm
                    answer = {
                        'type': 'action',
                        'location': [nr, nc],
                    }
                else:
                    answer = None

            elif msg["type"] == "end":
                games[msg["game"]].end_game()
                answer = None
            else:
                logger.error("Unknown message type:\n{}".format(msg))

            if answer is not None:
                await websocket.send(json.dumps(answer))
                logger.info("> {}".format(answer))
    except websockets.exceptions.ConnectionClosed as err:
        logger.info("Connection closed")
    logger.info("Exit handler")


def start_server(port):
    server = websockets.serve(handler, 'localhost', port)
    print("Running on ws://127.0.0.1:{}".format(port))
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()


# COMMAND LINE INTERFACE

def main(argv=None):
    global agentclass
    parser = argparse.ArgumentParser(
        description='Start agent to play Dots and Boxes')
    parser.add_argument('--verbose', '-v', action='count',
                        default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count',
                        default=1, help='Quiet output')
    parser.add_argument('port', metavar='PORT', type=int,
                        help='Port to use for server')
    args = parser.parse_args(argv)

    logger.setLevel(
        max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    agentclass = DotsAndBoxesAgent
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
