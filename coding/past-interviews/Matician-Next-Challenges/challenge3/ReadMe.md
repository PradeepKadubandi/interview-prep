## General Notes / Disclaimers
- I focused on solving the task at hand than some engineering aspects, For ex:
  - I did not use a CMake file or a project / IDE, I have one source file and used g++ from terminal to compile and then run (Used VS Code for editing). Arguably single file also saves time for reviewer and maybe preferred for the challenge (based on previous challenge asks). 
  - I did not focus on unit tests OR automated stress / functional tests, I can write these but need more time. Let me know if testing was initially inteded to be evaluated. (I did verify the functionality/stress manually using a tool, details below!)
  - To make the task easier, I started with this sample: https://github.com/chriskohlhoff/asio/blob/master/asio/src/examples/cpp17/coroutines_ts/chat_server.cpp and extended it to meet the requirements of challenge.
- Also I assumed that the task is to implement a single node server (based on 100 connections requirement) as opposed to a horizontally scalable, distributed chat server (which requires extending the solution to using a load balancer in front of a multi node cluster. Obviously the room management and message management needs to be coordinated among such multi node cluster and that would be a lot more complex task/project than this). We can brainstorm about ideas on how to do that if I move forward.
- I did not care too much about naming conventions and indentation / bracing standards. I followed what's in the reference sample to keep things consistent. These are easy to fix. For large existing projects, usually the best approach is to maintain consistency with existing code. Obviously in a team setting, coding standards are set and agreed upon by team.

## Compilation
- Platform:
  - I used my Mac to run and test. I only did tests on local machine in the interest of time (and also hardware availability).
  
- Dependencies:
  - asio (stand alone though this is included in boost as well): https://think-async.com/Asio/asio-1.18.0/doc/index.html
  - I also used boost libraries (string algorithms): https://www.boost.org/doc/libs/1_74_0/doc/html/string_algo.html
  - I did not include dependencies in the submission as there are a lot of files. I am assuming the reviewer can set the dependencies up for trying the code. I can share the executable or dependencies if needed.

- Command I used for compilation (requires asio and boost libraries in the mentioned path - can use bcp to only copy the specific boost library dependency chain):

``` 
g++ -std=c++17 -O1 chat_server.cpp -o bin/chat_server.exe -I lib/asio-1.18.0/include/ -fcoroutines-ts
```

## Stress Testing:
- Used the tool: https://github.com/satori-com/tcpkali
- Reference: https://github.com/satori-com/tcpkali/blob/master/doc/tcpkali.man.md
- Test 100 connections all joining 1 room and sending 10 messages per second for 30 seconds:

```
tcpkali -e -1 "join room1 User\{connection.uid}\r\n" -c 100 -T 30 -m "\{re [a-z0-9]+}\r\n" -r 10 localhost:1234
```

- Test 100 connections with 10 rooms sending 5 messages per second for 2 minutes:

```
tcpkali -e -1 "join room\{connection.uid%10} User\{connection.uid}\r\n" -c 100 -T 120 -m "\{re [a-z0-9]+}\r\n" -r 5 localhost:1234
```

- Above uses messages of average length 10 (short messages). Tried the same test with a very large message (~20 KB) repeatedly sent over the session (this uncovered an issue for me):

```
tcpkali -e -1 "join room\{connection.uid%10} User\{connection.uid}\r\n" -c 100 -T 120 -f large_message.txt -r 10 localhost:1234
```

- When I tried the above command for sending large message (~20 KB) for an hour, it eventually hung my machine. Read the known issues section for further comments.

- I did run 100 connections with 10 rooms and each participant sending 10 messages per second of size = 100 bytes (length 100 string in message.txt) for an hour and the local server is able to handle that load well (I was parallelly working on my MAC with 4 GB RAM when this is running). In this scenario, the resident set memory seemed to be under 50 MB.

- Validate server coalescing of packets (have another session with telnet with another user joined in room1 to observe the messages):
  - Breaking the message into 2 packets and combining the first packet with JOIN command packet

    ```
    tcpkali -e -1 "join room1 UserNameEnd\r\nShouldSeeThisInFirstMessage_And_" -1 "AlsoThis_In_Same_Line\r\n"  localhost:1234
    ```

  - Breaking the join command into several packets:

    ```
    tcpkali -e -1 "join ro" -1 "om1 UserName" -1 "End\r\nShouldSeeThisInFirstMessage_And_" -1 "AlsoThis_In_Same_Line\r\n"  localhost:1234
    ```

## Known Issues:
- There are no limits imposed in code for throttling clients. This means there could be enough load at which the system hangs. Right now, this happens when there are 100 client connections and 10 rooms (and each room with 10 participants) AND each client sends a 20 KB message every 0.1 second (10 per second).
  - There are some simple to sophisticated techniques to deal with this. At the simplest, we can deny client connections based on number of active clients / rooms. A little more sophisticated solution is to do that based on memory / cpu metrics of machine. And implementing thorttling of messages instead of denying client connections would be best user experience but requires more sophisticated implementation.
- Custom port code is there but was not working on my Mac when I tried it. Did not have chance to investigate it fully. (I was more focused on memory testing in various stress scenarios and making optimizations)

## Possible improvements:
- Server side logging:
  - basic metrics (like number of rooms) logged periodically
  - debug messages / error messages
