"""
Template for the Machine Learning Project for EE782
"""

import sys
import argparse
import logging
import asyncio
import websockets
import json
from collections import defaultdict
import random
import uuid
import time
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


def checkwinner(board):
    if 0 not in list(np.ravel(board)):
        return 0
    win = [3, -3]
    winners = {3: 1, -3: 2}
    for i in range(3):
        a = board[:, i].sum()
        if a in win:
            return winners[a]
    for i in range(3):
        a = board[i, :].sum()
        if a in win:
            return winners[a]
    a = sum(board[i, i] for i in range(3))
    if a in win:
        return winners[a]
    a = sum(board[i, 2-i] for i in range(3))
    if a in win:
        return winners[a]
    return None


def start_competition(address1, address2, episodes, noswap):
    episode = 0
    winners = []
    for episode in range(episodes):
        if not noswap:
            swap = random.random() > 0.5
            if swap:
                a1 = address2
                a2 = address1
            else:
                a1 = address1
                a2 = address2
            asyncio.get_event_loop().run_until_complete(
                connect_agent(a1, a2, winners, episode, swap))
        else:
            asyncio.get_event_loop().run_until_complete(
                connect_agent(address1, address2, winners, episode, 0))
        last_n = winners[-min(500, len(winners)):]
        print("Epsiode {} Cumulative Score: {} - {} - {}".format(
            episode, last_n.count(1), last_n.count(2), last_n.count(0)))
    print("Player 1 won {}, Player 2 won {} and Draw occured {} times".format(
        winners.count(1), winners.count(2), winners.count(0)))


async def connect_agent(uri1, uri2, winners, episode, sort_key):
    cur_game = str(uuid.uuid4())
    cur_player = 1
    winner = None
    board = np.zeros((3, 3))

    async with websockets.connect(uri1) as websocket1:
        async with websockets.connect(uri2) as websocket2:

            msg = {
                "type": "start",
                "episode": episode,
                "player": 1,
                "game": cur_game,
            }
            await websocket1.send(json.dumps(msg))
            msg["player"] = 2
            await websocket2.send(json.dumps(msg))

            while winner is None:
                if cur_player == 1:
                    msg = await websocket1.recv()
                else:
                    msg = await websocket2.recv()
                try:
                    msg = json.loads(msg)
                except json.decoder.JSONDecodeError as err:
                    logger.debug(err)
                    continue
                if msg["type"] != "action":
                    logger.error("Unknown message: {}".format(msg))
                    continue
                r, c = msg["location"]
                next_player = user_action(r, c, cur_player, board)
                winner = checkwinner(board)
                if winner is None:
                    msg = {
                        "type": "action",
                        "game": cur_game,
                        "player": cur_player,
                        "nextplayer": next_player,
                        "location": [r, c],
                    }
                    await websocket1.send(json.dumps(msg))
                    await websocket2.send(json.dumps(msg))

                cur_player = next_player

            msg = {
                "type": "end",
                "game": cur_game,
                "player": cur_player,
                "nextplayer": 0,
                "location": [r, c],
                "winner": winner
            }
            await websocket1.send(json.dumps(msg))
            await websocket2.send(json.dumps(msg))

    if sort_key == 0:
        winners.append(winner)
    else:
        winners.append({0: 0, 1: 2, 2: 1}[winner])
    return winners


def user_action(r, c, cur_player, board):
    assert r in [0, 1, 2]
    assert c in [0, 1, 2]
    board[r, c] = {1: 1, 2: -1}[cur_player]
    return 3 - cur_player


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Start agent to play Dots and Boxes')
    parser.add_argument('--verbose', '-v', action='count',
                        default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count',
                        default=0, help='Quiet output')
    parser.add_argument('--episodes', '-e', type=int,
                        default=200, help='Number of episodes')
    parser.add_argument('--noswap', action='store_true',
                        help='Dont swap players')
    parser.add_argument('agents', nargs=2, metavar='AGENT',
                        help='Websockets addresses for agents')
    args = parser.parse_args(argv)

    logger.setLevel(
        max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    start_competition(args.agents[0], args.agents[1],
                      args.episodes, args.noswap)


if __name__ == "__main__":
    sys.exit(main())
