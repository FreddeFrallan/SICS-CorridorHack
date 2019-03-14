from __future__ import print_function

import time, atexit
import multiprocessing as mp
from Network.Producer._Server import runProducerServer

'''
Will spawn an async server as a new process using the specified port.
Several clients can then join to subscribe to this producer and consume its content.
'''
class Node:

    def __init__(self, port):
        self.producerQ = mp.Queue(maxsize=0)
        self.p = mp.Process(target=runProducerServer, args=(port, self.producerQ))
        self.p.start()
        atexit.register(self._killNode)
        time.sleep(1)  # For some reason this seems to make a difference...

    def updateContent(self, content):
        self.producerQ.put(content)

    def _killNode(self):
        self.p.terminate()
        self.p.join()
