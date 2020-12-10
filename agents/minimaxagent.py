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
init = True
agent = None


class MiniMaxAgent:

    def __init__(self, player):
        self.player = player
        self.board = np.zeros((3, 3), dtype=np.int)
        self.table = {}

    def register_action(self, row, column, player):
        self.board[row][column] = {True: 1, False: -1}[self.player==player]

    def hash_fn(self, board):
        a = (board + 1).reshape(-1)
        b = [(3**i)*a[i] for i in range(9)]
        if self.player==2:
            return -sum(b)-1
        return sum(b)

    def check_over(self, board):
        win = {3: 1, -3: -1}
        ans = None
        for i in range(3):
            a = board[:, i].sum()
            if a in win.keys():
                ans = win[a]
        for i in range(3):
            a = board[i, :].sum()
            if a in win.keys():
                ans = win[a]
        a = sum(board[i, i] for i in range(3))
        if a in win.keys():
            ans = win[a]
        a = sum(board[i, 2-i] for i in range(3))
        if a in win.keys():
            ans = win[a]
        if ans==None and 0 not in list(board.reshape(-1)):
            ans = 0
        return ans

    def search(self, board, maximise):
        h = self.hash_fn(board)
        if h in self.table:
            return self.table[h]
        end = self.check_over(board)
        if end is not None:
            self.table[h] = (end, None)
            return end, None
        children = np.transpose(np.where(board == 0))
        if maximise:
            value = -np.inf
            for c in children:
                new_board = board.copy()
                new_board[c[0], c[1]] = 1
                a, _ = self.search(new_board, False)
                if a >= value:
                    value = a
                    move = c
            self.table[h] = (value, move)
            return value, move
        else:
            value = np.inf
            for c in children:
                new_board = board.copy()
                new_board[c[0], c[1]] = -1
                a, _ = self.search(new_board, True)
                if a <= value:
                    value = a
                    move = c
            self.table[h] = (value, move)
            return value, move

    def next_action(self):
        move = self.search(self.board.copy(), True)[1]
        if move is None:
            print(None)
            return
        r, c = move
        return int(r), int(c)

    def reset(self, player):
        self.player = player
        self.board = np.zeros((3, 3), dtype=np.int)


# MAIN EVENT LOOP
async def handler(websocket, path):
    global init, agent
    logger.info("Start listening")
    # msg = await websocket.recv()
    async for msg in websocket:
        logger.info("< {}".format(msg))
        msg = json.loads(msg)
        answer = None
        if msg["type"] == "start":
            if init:
                agent = MiniMaxAgent(msg["player"])
                init = False
            else:
                agent.reset(msg['player'])
            if msg["player"] == 1:
                nm = agent.next_action()
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
            agent.register_action(r, c, msg["player"])
            if msg["nextplayer"] == agent.player:
                nm = agent.next_action()
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
            r, c = msg["location"]
            agent.register_action(r, c, msg["player"])
            answer = None
        else:
            logger.error("Unknown message type:\n{}".format(msg))

        if answer is not None:
            await websocket.send(json.dumps(answer))
            logger.info("> {}".format(answer))
    logger.info("Exit handler")


def start_server(port):
    server = websockets.serve(handler, 'localhost', port)
    print("Running on ws://127.0.0.1:{}".format(port))
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()


# COMMAND LINE INTERFACE

def main(argv=None):
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

    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
