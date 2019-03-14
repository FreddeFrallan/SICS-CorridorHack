from Network.Consumer.Node import Node
import enum, threading

''' Simple Enum class for bookkeeping'''


class _AsyncNodeMode(enum.Enum):
    Automatic = 1,
    Manual = 2


'''
AsyncNode will fetch updates just like a normal consumer Node, but does this on a different thread.

To get the recent update use the method "getCurrentUpdate()"
DO NOT try to access the data directly from "self._currentUpdate" as this is not thread-safe.

To check if there's a new updates since last you can use the variable "hasNewUpdate"

They way fetches is done depends on the AsyncNodeMode.
Automatic:
    The Node is always looking for new updates and starts doing this at creation.
Manual:
    To start an asyncFetch call the "manualAsyncFetchUpdate".
'''


class AsyncNode:

    def __init__(self, ip, port, mode=_AsyncNodeMode.Automatic):
        self.Node = Node(ip, port)
        self.mode = mode

        self._currentUpdate = None
        self.hasNewUpdate = False
        self._updateLock = threading.Lock()

        if (self.mode == _AsyncNodeMode.Automatic):
            self._updateThread = threading.Thread(target=self._automaticAsyncFetchUpdate)
            self._updateThread.start()
        if (self.mode == _AsyncNodeMode.Manual):
            self._updateThread = threading.Thread(target=self.manualAsyncFetchUpdate)

    def manualAsyncFetchUpdate(self):
        if (self._updateThread.is_alive() == False):
            self._updateThread.run()

    def _automaticAsyncFetchUpdate(self):
        while (self.mode == _AsyncNodeMode.Automatic):
            self._asyncFetchUpdate()

    def _asyncFetchUpdate(self):
        update = self.Node.getUpdate()
        self._updateLock.acquire()
        try:
            self._currentUpdate = update
            self.hasNewUpdate = True
        finally:
            self._updateLock.release()

    def getCurrentUpdate(self):
        self._updateLock.acquire()
        try:
            tempUpdate = self._currentUpdate
            self.hasNewUpdate = False
        finally:
            self._updateLock.release()

        return tempUpdate
