# XO (TicTacToe) Leaning Environment

A framework for testing reinforcement learning algorithms on the simple XO game. Designed for learning and testing reinforcement learning and tree search algorithms. Based on similar [environment for dots and boxes](https://github.com/wannesm/dotsandboxes).

![Frontend](img/game.png)

## Usage
- #### Start the agents
```bash
$ python agents/minimaxagent.py 8081
$ python agents/dqnagent.py --test 8082
```
- #### Compete different agents
```bash
$ python xocompete.py ws://127.0.0.1:8081 ws://127.0.0.1:8082 --episodes 5000
```
- #### Start the GUI server
```bash
$ python xoserver.py 8080
```

## TODO
- [x] Write xocompete.py for playing 2 agents against each other.
- [x] Write a random agent.
- [x] Write  a xoserver.py and a frontend for human player.
- [x] Write the DQN.
- [x] Add convolutional architecture in DQN.
- [x] Write MiniMax agent.
- [x] Write AlphaBeta search tree agent.
- [ ] Write Dueling DQN agent.
- [ ] Let Agents tell their names while playing.


## Details
Course project for EE782, IIT Bombay.

### Team Members
- Mohd Safwan 17D070047
- Kumar Ashutosh 16D070043
- Manas Vashistha 17D070064

### Acknowledgement
- Prof. Amit Sethi
- E. Aakash for helping with the GUI. 
