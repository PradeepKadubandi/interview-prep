#include <cstdlib>
#include <deque>
#include <iostream>
#include <sstream>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <asio/awaitable.hpp>
#include <asio/detached.hpp>
#include <asio/co_spawn.hpp>
#include <asio/io_context.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/read_until.hpp>
#include <asio/redirect_error.hpp>
#include <asio/signal_set.hpp>
#include <asio/steady_timer.hpp>
#include <asio/use_awaitable.hpp>
#include <asio/write.hpp>
#include <boost/algorithm/string.hpp>

using asio::ip::tcp;
using asio::awaitable;
using asio::co_spawn;
using asio::detached;
using asio::redirect_error;
using asio::use_awaitable;
using namespace std;

//----------------------------------------------------------------------

/*
The reference sample used string objects for delivering messages 
to participants, for large messages, 
that's a lot of copying causing a lot of duplication and 
causing unnecessary active memory usage and compute.
While there are several ways to fix and improve it, 
I chose the quickest way (not probably best) to adapt the 
reference sample to the requirements by using shared_ptr objects 
for strings (avoiding duplication/copy of messages among 
different room participants.) Maybe using cstyle strings 
with careful memory management is a better long term choice. 
*/
typedef shared_ptr<string> string_ptr;

//----------------------------------------------------------------------

/*
Abstraction for chat room participant, 
chat_session will derive from this class and 
chat_room has a set of participants.
Was just done to break the circular reference (for compilation)
between chat_room and chat_session.
*/
class chat_participant
{
public:
  virtual ~chat_participant() {}
  virtual void deliver(string_ptr msg) = 0;
};

typedef shared_ptr<chat_participant> chat_participant_ptr;

//----------------------------------------------------------------------

/*
One instance per room. 
Handles participant management (joining and leaving room).
For broadcasting messages, just iterates over participants.
*/
class chat_room
{
public:
  void join(chat_participant_ptr participant)
  {
    participants_.insert(participant);
  }

  void leave(chat_participant_ptr participant)
  {
    participants_.erase(participant);
  }

  bool is_vacant()
  {
    return participants_.size() == 0;
  }

  void deliver(string_ptr msg)
  {
    for (auto participant: participants_)
      participant->deliver(msg);
  }

private:
  set<chat_participant_ptr> participants_;
};

typedef shared_ptr<chat_room> chat_room_ptr;

//----------------------------------------------------------------------

/*
One object per entire server process life time.
Responsible for room management (creating, removing room and adding participants).
Could be made singleton as a potential coding practice improvement.
*/
class chat_room_manager
{
  public:
    chat_room_ptr add_user_to_room(string room_name, chat_participant_ptr user)
    {
      if (rooms.find(room_name) == rooms.end())
      {
        mtx.lock();
        if (rooms.find(room_name) == rooms.end())
        {
          rooms[room_name] = make_shared<chat_room>();
          // If join was not done here, a race condition could cause exception : 
          // u1 : creates room which is about to be deleted, u1's critical section is called first and adds room,
          // remove's critical section gets next but deletes the room because join did not happen yet because room is vacant.
          // Now u1 holds a bad reference.
          rooms[room_name]->join(user); 
        }
        mtx.unlock();
      }
      else
      {
        rooms[room_name]->join(user);
      }
      return rooms[room_name];
    }
    void remove_room_if_vacant(string room_name)
    {
      if (rooms.find(room_name) != rooms.end() && rooms.at(room_name)->is_vacant())
      {
        mtx.lock();
        if (rooms.find(room_name) != rooms.end() && rooms.at(room_name)->is_vacant())
        {
          rooms.erase(room_name);
        }
        mtx.unlock();
      }
    }
  private:
    unordered_map<string, chat_room_ptr> rooms;
    mutex mtx;
};

//----------------------------------------------------------------------

/*
Manages one client session (so one per client session). 
Major part of the code. Reader reads and processes next message 
(first message is treated as join room command and 
later messages are broadcast messages). 
Writer writes to the client socket as long as there are messages, 
otherwise waits using an asynchronous timer.
*/
class chat_session
  : public chat_participant,
    public enable_shared_from_this<chat_session>
{
public:
  chat_session(tcp::socket socket, chat_room_manager& room_manager)
    : socket_(move(socket)),
      timer_(socket_.get_executor()),
      room_manager_(room_manager)
  {
    timer_.expires_at(chrono::steady_clock::time_point::max());
  }

  void start()
  {
    co_spawn(socket_.get_executor(),
        [self = shared_from_this()]{ return self->reader(); },
        detached);

    co_spawn(socket_.get_executor(),
        [self = shared_from_this()]{ return self->writer(); },
        detached);
  }

  void deliver(string_ptr msg)
  {
    write_msgs_.emplace_back(msg);
    timer_.cancel_one();
  }

private:
  // Processes first message and joins a room on valid command.
  void join_room(string& msg)
  {
    // We know that last character is a new line. 
    // Also handle when new line contains carriage return.
    msg.erase(msg.size()-1, 1);
    if (msg[msg.size()-1] == '\r')
    {
      msg.erase(msg.size()-1, 1);
    }

    vector<string> parts;
    boost::split(parts, msg, boost::is_space(), boost::algorithm::token_compress_on);

    if ((parts.size() > 3) || !boost::iequals(JOIN, parts[0]) || parts[1].size() > 20 || parts[2].size() > 20)
    {
      deliver(make_shared<string>(ERROR));
      shouldStop = true;
    } 
    else
    {
      room_name = parts[1];
      user_name = parts[2];
      room_ = room_manager_.add_user_to_room(parts[1], shared_from_this());
      string_ptr sent_msg = make_shared<string>(user_name + JOINED_SUFFIX);
      room_->deliver(sent_msg);
    }
  }

  awaitable<void> reader()
  {
    try
    {
      for (string read_msg;;)
      {
        size_t n = co_await asio::async_read_until(socket_,
            asio::dynamic_buffer(read_msg, 20000), "\n", use_awaitable);

        auto msg = read_msg.substr(0, n);
        read_msg.erase(0, n);

        if (room_ != NULL)
        {
          string_ptr sent_msg = make_shared<string>(user_name + MESSAGE_SEPARATOR + msg);
          room_->deliver(sent_msg);
        } 
        else 
        {
          join_room(msg);
        }
      }
    }
    catch (exception& e)
    {
      deliver(make_shared<string>(ERROR));
      shouldStop = true;
    }
  }

  awaitable<void> writer()
  {
    try
    {
      while (socket_.is_open())
      {
        if (write_msgs_.empty())
        {
          asio::error_code ec;
          co_await timer_.async_wait(redirect_error(use_awaitable, ec));
        }
        else
        {
          auto msg = *write_msgs_.front();
          auto ignored = co_await asio::async_write(socket_,
              asio::buffer(msg), use_awaitable);

          write_msgs_.pop_front();

          if (shouldStop)
          {
            stop();
          }
        }
      }
    }
    catch (exception&)
    {
      stop();
    }
  }

  void stop()
  {
    if (room_ != NULL) {
      string_ptr sent_msg = make_shared<string>(user_name + LEFT_SUFFIX);
      room_->deliver(sent_msg);
      room_manager_.remove_room_if_vacant(room_name);
    }
    socket_.close();
    timer_.cancel();
  }

  // Construction time members.
  tcp::socket socket_;
  asio::steady_timer timer_;
  chat_room_manager& room_manager_;

  // Members initialized when joining a room.
  chat_room_ptr room_;
  string user_name;
  string room_name;
  bool shouldStop = false;

  // Holds pointers for messages to be sent to this session.
  deque<string_ptr> write_msgs_;

  // Constants - ideally these could be in another constants class.
  // I would have done it if I had split the code into multiple files and headers.
  static inline const string JOIN = "join";
  static inline const string JOINED_SUFFIX = " has joined\n";
  static inline const string LEFT_SUFFIX = " has left\n";
  static inline const string MESSAGE_SEPARATOR = ": ";
  static inline const string ERROR = "Error\n";
};


//----------------------------------------------------------------------

/*
Method handler for accepting connections.
Initiates client connections in a separate strand.
*/
awaitable<void> listener(tcp::acceptor acceptor)
{
  chat_room_manager room_manager;

  for (;;)
  {
    make_shared<chat_session>(
        co_await acceptor.async_accept(use_awaitable),
        room_manager
      )->start();
  }
}

//----------------------------------------------------------------------

int main(int argc, char* argv[])
{
  try
  {
    unsigned short port = 1234;
    if (argc == 2)
    {
      port = atoi(argv[2]);
    }

    asio::io_context io_context;

    co_spawn(io_context,
        listener(tcp::acceptor(io_context, {tcp::v4(), port})),
        detached);

    asio::signal_set signals(io_context, SIGINT, SIGTERM);
    signals.async_wait([&](auto, auto){ io_context.stop(); });

    io_context.run();
  }
  catch (exception& e)
  {
    cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}