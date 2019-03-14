from twisted.internet import reactor
from twisted.internet.protocol import Factory, connectionDone
from twisted.protocols.basic import LineReceiver
import Network.Contants as C
import threading

# Bookkeeping data
CONTENT_LOCK = threading.Lock()
CURRENT_CONTENT = None
CONTENT_CONSUMER_QUEUE = []


class ProducerServer(LineReceiver):

    def connectionMade(self):
        CONTENT_LOCK.acquire()
        try:
            CONTENT_CONSUMER_QUEUE.append(self)
        finally:
            CONTENT_LOCK.release()

        print('Connection made from {}'.format(self.transport.getPeer()))

    # When connection is lost we remove our selves from the consumer Queue
    def connectionLost(self, reason=connectionDone):
        CONTENT_LOCK.acquire()
        try:
            CONTENT_CONSUMER_QUEUE.remove(self)
        finally:
            CONTENT_LOCK.release()

    def _sendUpdateToClient(self, content):
        self.transport.write(content)


# Separate thread that keeps listening to the Producer Node that spawned this server.
def contentUpdater(producerQ):
    global CURRENT_CONTENT, CONTENT_LOCK, CONTENT_VERSION

    while (True):
        newContent = producerQ.get()
        while (producerQ.empty() == False):  # Barbaric way of flushing until getting the most recent update...
            newContent = producerQ.get()

        CONTENT_LOCK.acquire()
        try:
            CURRENT_CONTENT = newContent
            sendUpdateToClients(CURRENT_CONTENT)
        finally:
            CONTENT_LOCK.release()


def encodeMsg(content):
    sizeInBytes = len(content).to_bytes(C.HEADER_BYTE_SIZE, C.HEADER_ENDIAN_TYPE)
    return sizeInBytes + content


# Should only be called when holding CONTENT_LOCK
def sendUpdateToClients(content):
    content = encodeMsg(content)
    lostClients = []
    for c in CONTENT_CONSUMER_QUEUE:
        try:
            reactor.callFromThread(c._sendUpdateToClient, content)
        except:
            lostClients.append(c)

    for c in lostClients:
        CONTENT_CONSUMER_QUEUE.remove(c)


# Called from spawning producer node
def runProducerServer(port, listenQ):
    contentThread = threading.Thread(target=contentUpdater, args=(listenQ,))
    contentThread.start()

    factory = Factory()
    factory.protocol = ProducerServer
    reactor.listenTCP(port, factory)
    reactor.run()
