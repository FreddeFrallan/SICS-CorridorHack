Lightweight network Producer-Consumer mini-framework, intended as a starting point in the discussion of the notorious “corridor hack”.

  

Have a look in the Demo folder for a brief demonstration of how to setup a video producer and consumer.

  
\
\
In short there exists two types of nodes:

### Producers (Server)

 - Allows multiple clients to connect over TCP and will broadcast to all clients whenever its content is updated.


    

### Consumer (Client)

 - Connects to the server using TCP and continuously reads and parses the received data stream into content updates.
 -  Can be accessed either by the blocking or asynchronous interface.

  \
  \
For now there’s no way to tell what type of nodes that exists on the network or what protocol a certain node is using. All of this has to be currently be hardcoded into the application!

  

Discussed future works:

### Producer-Hub

 - Keeping track of all the current Producers on the network, allowing consumers to easily find available producers on the network.

-   Keeping track of what protocol certain nodes use.
    

### External Network Configuration
    
-   Storing all of the Ports & IP-addresses producers in an external file
