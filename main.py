from Chessnut import Game

from subprocess import Popen, PIPE
from threading import Thread
from queue import Queue, Empty

import atexit
import os
import sys
import inspect

from Chessnut import Game

# Code from this notebook: https://www.kaggle.com/code/ashketchum/an-example-c-bot-outputs-a-randomly-chosen-move

my_agent_process = None
t = None
q = None

def cleanup_process():
    global my_agent_process
    if my_agent_process is not None:
        my_agent_process.kill()

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

# The last def in the main is the one called by the Kaggle game runner
# Popen allows to write to stdin and read from stdout of cpp agent
# Thread reads from stdout cpp

def cpp_agent(observation):
    global my_agent_process, t, q

    agent_process = my_agent_process

    ### Do not edit ###
    if agent_process is None:
        src_file_path = inspect.getfile(lambda: None)  # the path to this main.py file. https://stackoverflow.com/a/53293924
        cwd = os.path.split(src_file_path)[0]
        agent_process = Popen(["./my_chess_bot.out"], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd)
        my_agent_process = agent_process
        atexit.register(cleanup_process)

        # following 4 lines from https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python
        q = Queue()
        t = Thread(target=enqueue_output, args=(agent_process.stderr, q))
        t.daemon = True  # thread dies with the program
        t.start()

    # read observation, and, send inputs to our cpp agent in required format
    # Read observation:
    game = Game(observation.board)
    moves = list(game.get_moves())

    # Our cpp bot expects:
    # first line: current player color (0 for white, 1 for black)
    # second line: board FEN string
    # third line: space separated available moves
    # Form the input to our cpp agent:
    input_to_cpp_agent = f"{0 if game.state.player == 'w' else 1}\n"  # first line
    input_to_cpp_agent += f"{game.get_fen()}\n"  # add second line
    input_to_cpp_agent += f"{' '.join(moves)}\n"  # add third line
    # send it to the cpp agent
    agent_process.stdin.write(input_to_cpp_agent.encode())
    agent_process.stdin.flush()

    # wait for data written to stdout
    agent_res = (agent_process.stdout.readline()).decode()
    agent_res = agent_res.strip()  # trim any whitespace in the outputted move

    # get the data printed to err stream of c++ agent, and print to the err stream of this python process.
    # remember that a thread running the function "enqueue_output" is constantly reading err stream of the c++ agent
    # and putting the data in a queue named "q". Here, we just get from that queue.

    while True:
        try:
            line = q.get_nowait()
        except Empty:
            # no standard error received, break
            break
        else:
            # standard error output received, print it out
            print(line.decode(), file=sys.stderr, end='')

    # return the move
    return agent_res