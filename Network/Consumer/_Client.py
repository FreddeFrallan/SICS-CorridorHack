from twisted.internet import reactor
from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.protocols.basic import LineReceiver
import Network.Contants as C

# Lazily created global queue, to easily communicate with the Twisted Client.
# This would not be needed if a better framework was used for the client....
SEND_TO_NODE_QUEUE = None


# Simple Twisted client, could just as easily have been created using any other TCP-based framework or raw TCP.
class ConsumerClient(LineReceiver):
    # Variables connected to parsing the incoming TCP stream into packages
    currentPackage = b''
    bufferingMsg = False
    currentMsgSize = 0

    def connectionMade(self):
        print('Connection made from {}'.format(self.transport.getPeer()))

    # Called when there exists data on the TCP stream. For now the messages encode the byte size in the 4 first bytes.
    def dataReceived(self, data):
        if (self.bufferingMsg == False):
            data = self.currentPackage + data

            # If we're given less bytes than the header, we're yet unable to decode the intended messages size.
            if (len(data) < C.HEADER_BYTE_SIZE):
                self.currentPackage = data
                return

            # Decode the msg header containing the message size, keeping the rest of the message for the data.
            self.bufferingMsg = True
            self.currentMsgSize = int.from_bytes(data[:C.HEADER_BYTE_SIZE], C.HEADER_ENDIAN_TYPE)
            self.currentPackage = data[C.HEADER_BYTE_SIZE:]
        else:
            self.currentPackage += data

        # Once we have collected enough bytes for our message, we send it to the ConsumerNode
        # Recursively handling any leftover bytes
        if (len(self.currentPackage) >= self.currentMsgSize):
            SEND_TO_NODE_QUEUE.put(self.currentPackage[:self.currentMsgSize])
            #print("Got grame")
            self.bufferingMsg = False

            restMsg = self.currentPackage[self.currentMsgSize:]
            self.currentPackage = b''
            if (len(restMsg) > 0):
                self.dataReceived(restMsg)


# Called from spawning consumer node
def runConsumerClient(ip, port, sendQ):
    global SEND_TO_NODE_QUEUE
    SEND_TO_NODE_QUEUE = sendQ

    point = TCP4ClientEndpoint(reactor, ip, port)
    connectProtocol(point, ConsumerClient())
    reactor.run()
