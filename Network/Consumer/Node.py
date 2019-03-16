import atexit, time
import multiprocessing as mp
from Network.Consumer._Client import runConsumerClient


class Node:
    '''
    Tries to connect to the specified server upon creation, once this connection is established you can get the current
    update by calling "getUpdate()". This is a blocking call, if you wish to make async updates use the AsyncNode.

    This consumer node could have been using raw TCP without any of the Twisted API. The only thing required from the
    consumer is that it uses TCP to connect to Socket.
    '''

    def __init__(self, ip, port):
        self.toNodeQ = mp.Queue(maxsize=0)
        self.p = mp.Process(target=runConsumerClient, args=(ip, port, self.toNodeQ))
        self.p.start()

        atexit.register(self._killNode)
        time.sleep(1)

    def getUpdate(self):  # Blocking call
        update = self.toNodeQ.get()
        while (self.toNodeQ.qsize() > 0):  # Barbaric way of getting the last update
            update = self.toNodeQ.get()
        return update

    def _killNode(self):
        self.p.terminate()
        self.p.join()
