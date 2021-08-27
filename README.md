# XO (TicTacToe) Learning Environment

A framework for testing reinforcement learning algorithms on the simple XO game. Designed for learning and testing reinforcement learning and tree search algorithms. Based on similar [environment for dots and boxes](https://github.com/wannesm/dotsandboxes).

![Frontend](img/game.png)

The GUI live demo is at [safwankdb.github.io/xo](https://safwankdb.github.io/xo)

## Usage

- #### Start the agents
This is the program that runs a game-playing agent. This application listens to [websocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API) requests that communicate game information and sends back the next action it wants to play.
```bash
$ python agents/youragent.py 8080
$ python agents/minimaxagent.py 8081
$ python agents/dqnagent.py --test 8082
```
This starts a websocket on the given port that can receveive JSON messages. The JSON messages given below should be handled by your agent.

- #### Compete different agents
```bash
$ python xocompete.py ws://127.0.0.1:8081 ws://127.0.0.1:8082 --episodes 5000
```
- #### Start the GUI server
```bash
$ python xoserver.py 8080
```


- ### Communicating with the game

Both players get a message that a new game has started:
```
{
    "type": "start",
    "player": 1,
    "game": "123456"
}
```
where `player` is the number assigned to this agent.

If you are player 1, reply with the first action you want to perform:
```
{
    "type": "action",
    "location": [1, 1],
}
```
The field `location` is expressed as row and column (zero-based numbering).

When an action is played, the message sent to both players is:
```
{
    "type": "action",
    "game": "123456",
    "player": 1,
    "nextplayer": 2,
    "location": [1, 1],
}
```

If it is your turn you should answer with a message that states your next
move:
```
{
    "type": "action",
    "location": [1, 1],
}
```
When the game ends after an action, the message is slightly altered:
```
{
    "type": "end",
    "game": "123456",
    "player": 1,
    "nextplayer": 0,
    "location": [1, 1],
    "winner": 1
}
```
The `type` field becomes `end` and a new field `winner` is set to the player
that has won the game.

### Provided Agents

- **randomagent**: Chooses a move randomly from all valid moves.
- **simpleagent**: Chooses the move with lowest index among valid moves.
- **minimaxagent**: Performs a full depth minimax tree search to find best moves, it can never lose.
- **alphabetaagent**: Uses alpha beta pruning in tree search and finds sub-optimal moves.
- **dqnagent**: Uses Deep Q Network to approximate the Q function and learns to play online.
  
<center>
  
|    Player 1 / Player 2       |  MiniMax  |  AlphaBeta  |    Random   |   Simple   |
|:---------:|:---------:|:-----------:|:-----------:|:----------:|
|  **MiniMax**  |  0-0-1000 |   1000-0-0  |   989-0-11  |  1000-0-0  |
| **AlphaBeta** |  0-1000-0 |   1000-0-0  |  859-88-53  |  1000-0-0  |
|   **Random**  | 0-815-185 | 208-598-194 | 582-310-108 | 545-427-28 |
|   **Simple**  |  0-1000-0 |  765-180-55 |   0-1000-0  |  1000-0-0  |

Number of games won/lost/drawn per 1000 games.
</center>


## TODO
- [x] Host the xoserver somewhere.
- [x] Write xocompete.py for playing 2 agents against each other.
- [x] Write a random agent.
- [x] Write  a xoserver.py and a frontend for human player.
- [x] Write the DQN.
- [x] Add convolutional architecture in DQN.
- [x] Write MiniMax agent.
- [x] Write AlphaBeta search tree agent.
- [ ] Write a simple Q learning agent.
- [ ] Write a SARSA agent.  
- [ ] Write Dueling DQN agent.
- [ ] Let Agents tell their names while playing.

