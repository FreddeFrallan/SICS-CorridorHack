""" Announcing and locating a service """

import socket
import atexit
import multiprocessing
import time
import pickle
import zeroconf


def get_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('1.2.3.4', 1))  # dummy connect
        return s.getsockname()[0]

def register(name, port, properties={}):
    type_="_http._tcp.local."
    
    info = zeroconf.ServiceInfo(
        type_="_http._tcp.local.",
        name=name + '.' + type_,
        address=socket.inet_aton(get_ip()),
        port=port,
        weight=0,
        priority=0,
        properties=properties,
        server="kalleanka.local.")

    zc = zeroconf.Zeroconf()
    zc.register_service(info)

    @atexit.register
    def unreg():
        zc.unregister_service(info)
        zc.close()
        print('unregistered zeroconf')


def look_for(service_name, timeout=None, get_many=True, type_="_http._tcp.local."):
    services = []
    Added = zeroconf.ServiceStateChange.Added # stupid but necessary

    # semaphore used for synchronization
    # listen for just first one, or for many until timeout
    handler_done = multiprocessing.Event()

    def on_service_state_change(zeroconf, service_type, name, state_change):
        nonlocal services, handler_done
        if state_change is Added:
            info = zeroconf.get_service_info(service_type, name)
            if name.startswith(service_name):
                address = socket.inet_ntoa(info.address)
                port = info.port
                if get_many or len(services)==0:
                    services.append([name, address, port])
                if not get_many:
                    handler_done.set()

    # register a listner for zeroconf events
    zc = zeroconf.Zeroconf()
    zeroconf.ServiceBrowser(zc, type_=type_, handlers=[on_service_state_change])

    # wait until the listener found what we are looking for
    handler_done.wait(timeout)
    zc.close()
    return services

######################################################################
## Example usages


def send_cam(name, port, cam_id=0):
    import cv2
    import atexit
    import Network.Producer.Node
    import hello
    
    
    p = Network.Producer.Node.Node(port)
    hello.register(name, port)

    try:
        import pafy
        url = pafy.new(cam_id).getbest().url
        assert url.startswith('http')
        cam_id = url
    except:
        pass

    while True:
        cam = cv2.VideoCapture(cam_id)
        atexit.register(cam.release)

        print(f'Starting steam {cam.isOpened()}')
        while cam.isOpened():
            ok, img = cam.read()

            if ok:
                data = pickle.dumps(img[::3, ::3])
                p.updateContent(data)
            else:
                break

        cam.release()

        print('video stream ended')
        
    # NUKE IT!
    #import os; os._exit(0)

        
    
def browse_cam(name, timeout):
    import hello
    print(f'Available services starting with {name}:')
    for n, ip, port in hello.look_for(name, get_many=True, timeout=timeout):
        print(n)
    print()


def view_cam(name):
    import cv2
    import Network.Consumer.Node
    import hello

    [[_, ip, port]] = hello.look_for(name, get_many=False)
    c = Network.Consumer.Node.Node(ip, port)

    done = False
    while not done:
        img = pickle.loads(c.getUpdate())
        cv2.imshow('Frame', img)
        key = chr(cv2.waitKey(20) & 0xff)
        if key == 'q': done=True
        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    import random

    cmd = sys.argv[1]
    name = sys.argv[2]

    if cmd == 'browse':
        browse_cam(name, timeout=2)

    elif cmd == 'client':
        view_cam(name)

    elif cmd == 'server':
        # send a video device stream (nr), or from a file (name), or youtube url
        try:
            channel = 0
            channel = sys.argv[3]
            channel = int(channel)
        except: pass
        
        port = random.randint(0, 1000)+20000

        send_cam(name, port, channel)

    """
    On one machine do `python hello.py server myvideo`
    and on another do `python hello.py client myvideo`

    The client will (hopefully) find the ip addres and port of the server.
    """

