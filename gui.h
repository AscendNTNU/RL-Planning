#ifndef SIM_GUI_HEADER_INCLUDE
#define SIM_GUI_HEADER_INCLUDE
#define SIM_CLIENT_LISTEN_PORT 22334
#define SIM_SERVER_LISTEN_PORT 22335
#ifdef SIM_CLIENT_CODE
#define RECV_PORT SIM_CLIENT_LISTEN_PORT
#define SEND_PORT SIM_SERVER_LISTEN_PORT
#else
#define RECV_PORT SIM_SERVER_LISTEN_PORT
#define SEND_PORT SIM_CLIENT_LISTEN_PORT
#endif

// The simulator uses UDP packets for interprocess
// communication. Internally, the simulator polls
// for UDP packets containing a high-level command
// that the drone will execute. Furthermore, the
// simulator sends the complete world state at a
// predefined rate.

// To get state from the simulator and send drone
// commands to the simulator, do the following:
// #define SIM_CLIENT_CODE in your application,
// and include this file.

// The functions
//  - sim_init_msgs
//  - sim_recv_state
//  - sim_send_cmd
// can be used to communicate with the simulator.

// sim_recv_state will read as many packets as are
// available to acquire the latest one. This is to
// avoid the pipes clogging up if your loop is
// slower than the sending rate.

#define UDP_IMPLEMENTATION
#include "udp.h"

void sim_init_msgs(bool blocking)
{
    udp_open(RECV_PORT, !blocking);
}

#ifdef SIM_CLIENT_CODE
bool sim_recv_state(sim_State *result)
{
    sim_State buffer = {};
    return udp_read_all((char*)result, (char*)&buffer, sizeof(sim_State), 0);
}

void sim_send_cmd(sim_Command *cmd)
{
    udp_addr dst = { 127, 0, 0, 1, SEND_PORT };
    udp_send((char*)cmd, sizeof(sim_Command), dst);
}
#else
bool sim_recv_cmd(sim_Command *result)
{
    sim_Command buffer = {};
    return udp_read_all((char*)result, (char*)&buffer, sizeof(sim_Command), 0);
}

void sim_send_state(sim_State *state)
{
    udp_addr dst = { 127, 0, 0, 1, SEND_PORT };
    udp_send((char*)state, sizeof(sim_State), dst);
}
#endif

#endif
