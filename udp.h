// UDP socket wrapper for OSX, Linux and Windows
// Provides basic socket functionality of opening and closing a socket,
// and sending and receiving sized raw blocks of memory.
//
// http://www.ascendntnu.no/
//
// How to compile
// --------------
// This file contains both the header file and the implementation file.
// To compile, insert the following in A SINGLE source file in your project
//
//     #define UDP_IMPLEMENTATION
//     #include "udp.h"
//
// You may otherwise include this file as you would include a traditional
// header file. You can define UDP_ASSERT before the include to avoid using
// assert.h
//
// Changelog
// --------------
//   1.05 (15. jan 2016) send and recv take void* instead of char*
//                       to avoid unecessary conversions in user-level code.
//                       Added explanatory notes for the functions.
//                       Added udp_open_ip which takes an explicit IP to
//                       bind to.
//   1.02 (16. dec 2015) Discovered bug in udp_recv_all when using blocking
//                       sockets. Added preliminary fix.
//   1.00 (15. nov 2015) Added missing include for memcpy on linux
//
// Licence
// --------------
// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy,
// distribute, and modify this file as you see fit.
//
// No warranty for any purpose is expressed or implied by the author (nor
// by Ascend NTNU). Report bugs and send enhancements to the author.
//

#ifndef UDP_HEADER_INCLUDE
#define UDP_HEADER_INCLUDE
#include <stdint.h>
#include <cstring> // for memcpy
#ifndef UDP_ASSERT
#include <cassert>
#define UDP_ASSERT assert
#endif

struct udp_addr
{
    uint8_t ip0, ip1, ip2, ip3; // represents ip0.ip1.ip2.ip3
    uint16_t port;
};

// Your computer may have multiple network adapters, each with
// its own IP address. Because of this I have two functions,
// udp_open_ip and udp_open. If you care about which adapter
// is used in the binding process, use the former and specify
// a valid port *and* ip pair.

bool udp_open_ip(udp_addr addr, bool non_blocking);
bool udp_open(uint16_t port, bool non_blocking);
void udp_close();

// data, size
//   Before calling this function you need to prepare a
//   buffer that is big enough to hold size number of bytes.
// src
//   If nonzero, the struct pointed to by src is filled with
//   the IP address and the port number of the UDP connection
//   that sent the data.
// Return
//   Number of bytes read.
int  udp_recv(void *data, uint32_t size, udp_addr *src);

// data, size
//   A pointer to a block of contiguous memory of size bytes.
// dst
//   The IP address and port number of the receiving client socket.
// Return
//   Number of bytes sent.
int  udp_send(void *data, uint32_t size, udp_addr dst);

// result, buffer, size
//   You allocate these. They must both be atleast size bytes big.
// src
//   If nonzero, the struct pointed to by src is filled with
//   the IP address and the port number of the UDP connection
//   that sent the data.
// Return
//   true if atleast one packet with size number of bytes was
//   received, with the most recent packet stored in the buffer
//   pointed to by result.
//   false if no packets were read.
// Remarks
//   This function will attempt to exhaust the pending packets
//   for the bound socket, and return the last valid packet.
//   The function does not play well with a blocking socket,
//   especially if the rate at which this function is called
//   is lower than the rate of incoming packets. In that case,
//   packets will pile up, and you will get stale data.
bool udp_read_all(void *result, void *buffer,
                  uint32_t size, udp_addr *src);

#ifdef UDP_IMPLEMENTATION

#if defined(__linux) || defined(__APPLE__)
#include <netdb.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

static int udp_socket = 0;
static int udp_is_blocking = 0;

bool udp_open_ip(udp_addr addr, bool non_blocking)
{
    udp_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_socket < 0)
    {
        // Failed to open socket
        UDP_ASSERT(false);
        return false;
    }

    struct sockaddr_in address = {};
    address.sin_family = AF_INET;
    if (addr.ip0 == 0 &&
        addr.ip1 == 0 &&
        addr.ip2 == 0 &&
        addr.ip3 == 0)
        address.sin_addr.s_addr = INADDR_ANY;
    else
        address.sin_addr.s_addr = htonl(
        (addr.ip0 << 24) |
        (addr.ip1 << 16) |
        (addr.ip2 <<  8) |
        (addr.ip3));
    address.sin_port = htons(addr.port);

    if (bind(udp_socket, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        // Failed to bind socket
        UDP_ASSERT(false);
        return false;
    }

    if (non_blocking)
    {
        udp_is_blocking = 0;
        int opt = 1;
        if (ioctl(udp_socket, FIONBIO, &opt) == -1)
        {
            // Failed to set socket to non-blocking
            UDP_ASSERT(false);
            return false;
        }
    }
    else
    {
        udp_is_blocking = 1;
    }

    return true;
}

int udp_recv(void *data, uint32_t max_size, udp_addr *src)
{
    if (!udp_socket)
    {
        // Socket not initialized
        UDP_ASSERT(false);
    }

    struct sockaddr_in from;
    socklen_t from_length = sizeof(from);
    int bytes_read = recvfrom(
        udp_socket, data, max_size, 0,
        (struct sockaddr*)&from, &from_length);
    if (bytes_read <= 0)
        return 0;

    uint32_t from_address = ntohl(from.sin_addr.s_addr);
    if (src)
    {
        src->ip0  = (from_address >> 24) & 0xff;
        src->ip1  = (from_address >> 16) & 0xff;
        src->ip2  = (from_address >>  8) & 0xff;
        src->ip3  = (from_address >>  0) & 0xff;
        src->port = ntohs(from.sin_port);
    }

    return bytes_read;
}

int udp_send(void *data, uint32_t size, udp_addr dst)
{
    if (!udp_socket)
    {
        // Socket not initialized
        UDP_ASSERT(false);
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(
        (dst.ip0 << 24) |
        (dst.ip1 << 16) |
        (dst.ip2 <<  8) |
        (dst.ip3));
    address.sin_port = htons(dst.port);

    int bytes_sent = sendto(udp_socket, data, size, 0,
        (struct sockaddr*)&address, sizeof(struct sockaddr_in));

    return bytes_sent;
}

void udp_close()
{
    // Nothing to do here!
}

#elif _WIN32
#include <winsock2.h>
#pragma comment(lib, "wsock32.lib")
static uint32_t udp_socket = 0;
static int udp_is_blocking = 0;

bool udp_open_ip(udp_addr addr, bool non_blocking)
{
    WSADATA WsaData;
    if (WSAStartup(MAKEWORD(2, 2), &WsaData) != NO_ERROR)
    {
        // Windows failure
        UDP_ASSERT(false);
        return false;
    }

    udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp_socket <= 0)
    {
        // Failed to create socket
        UDP_ASSERT(false);
        return false;
    }

    // Bind socket to a port
    sockaddr_in address;
    address.sin_family = AF_INET;
    if (addr.ip0 == 0 &&
        addr.ip1 == 0 &&
        addr.ip2 == 0 &&
        addr.ip3 == 0)
        address.sin_addr.s_addr = INADDR_ANY;
    else
        address.sin_addr.s_addr = htonl(
        (addr.ip0 << 24) |
        (addr.ip1 << 16) |
        (addr.ip2 <<  8) |
        (addr.ip3));
    address.sin_port = htons(addr.port);
    if (bind(udp_socket, (const sockaddr*)&address, sizeof(sockaddr_in)) < 0)
    {
        // Failed to bind socket (maybe port was taken?)
        UDP_ASSERT(false);
        return false;
    }

    if (non_blocking)
    {
        // Set port to not block when calling recvfrom
        udp_is_blocking = 0;
        DWORD non_blocking = 1;
        if (ioctlsocket(udp_socket, FIONBIO, &non_blocking) != 0)
        {
            // Failed to set port to non-blocking
            UDP_ASSERT(false);
            return false;
        }
    }
    else
    {
        udp_is_blocking = 1;
    }

    return true;
}

int udp_recv(void *data, uint32_t size, udp_addr *src)
{
    if (!udp_socket)
    {
        // Socket not initialized
        UDP_ASSERT(false);
    }

    sockaddr_in from;
    int from_length = sizeof(from);
    int bytes_read = recvfrom(
        udp_socket, (char*)data, size, 0, (sockaddr*)&from, &from_length);

    if (bytes_read <= 0)
        return 0;

    uint32_t from_address = ntohl(from.sin_addr.s_addr);
    if (src)
    {
        src->ip0 = (from_address >> 24) & 0xff;
        src->ip1 = (from_address >> 16) & 0xff;
        src->ip2 = (from_address >>  8) & 0xff;
        src->ip3 = (from_address >>  0) & 0xff;
        src->port = ntohs(from.sin_port);
    }
    return bytes_read;
}

int udp_send(void *data, uint32_t size, udp_addr dst)
{
    if (!udp_socket)
    {
        // Socket not initialized
        UDP_ASSERT(false);
    }

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(
        (dst.ip0 << 24) |
        (dst.ip1 << 16) |
        (dst.ip2 <<  8) |
        (dst.ip3));
    address.sin_port = htons(dst.port);
    int bytes_sent = sendto(udp_socket, (char*)data, size,
        0, (sockaddr*)&address, sizeof(sockaddr_in));
    return bytes_sent;
}

void udp_close()
{
    WSACleanup();
}
#endif

bool udp_open(uint16_t port, bool non_blocking)
{
    udp_addr addr;
    addr.ip0 = 0;
    addr.ip1 = 0;
    addr.ip2 = 0;
    addr.ip3 = 0;
    addr.port = port;
    return udp_open_ip(addr, non_blocking);
}

bool udp_read_all(void *result,
                  void *buffer,
                  uint32_t size,
                  udp_addr *src)
{
    if (udp_is_blocking)
    {
        // I haven't implemented read_all correctly
        // for blocking sockets. Ideally, we would
        // try to read until we get blocket, at which
        // point we return the latest data. For now,
        // I just read once.

        // The implication of this is if your app
        // is receiving packets faster than it tries
        // to read_all them. In that case, packets
        // will pile up, and your app will read stale
        // data.
        uint32_t read_bytes = udp_recv(buffer, size, src);
        if (read_bytes != size)
        {
            return false;
        }
        else
        {
            memcpy(result, buffer, size);
            return true;
        }
    }

    uint32_t read_bytes = udp_recv(buffer, size, src);
    if (read_bytes != size)
    {
        return false;
    }
    else
    {
        memcpy(result, buffer, size);
        bool reading = true;
        while (reading)
        {
            read_bytes = udp_recv(buffer, size, src);
            if (read_bytes == size)
                memcpy(result, buffer, size);
            else
                reading = false;
        }
        return true;
    }
}

#endif // UDP_IMPLEMENTATION
#endif // UDP_HEADER_INCLUDE
