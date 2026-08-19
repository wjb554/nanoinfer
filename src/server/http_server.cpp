/// Concurrent + streaming HTTP server using Winsock2 (Windows) or POSIX
/// sockets.  Accept loop spawns a detached std::thread per connection; each
/// connection reads the full request, runs the handler, then closes.

#include "nanoinfer/server/http_server.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#define closesocket close
#endif

#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>

namespace nanoinfer {
namespace server {

namespace {

// ============================================================================
// Request parsing
// ============================================================================
// Parse HTTP request from raw bytes (same semantics as the original server).
HttpRequest parse_request(const char* data, int len) {
    HttpRequest req;
    std::string raw(data, len);

    // Parse first line: "POST /path HTTP/1.1"
    size_t nl = raw.find("\r\n");
    if (nl == std::string::npos) return req;
    std::string first = raw.substr(0, nl);

    size_t sp1 = first.find(' ');
    size_t sp2 = first.find(' ', sp1 + 1);
    if (sp1 != std::string::npos) req.method = first.substr(0, sp1);
    if (sp1 != std::string::npos && sp2 != std::string::npos)
        req.path = first.substr(sp1 + 1, sp2 - sp1 - 1);

    // Find body (after \r\n\r\n)
    size_t body_start = raw.find("\r\n\r\n");
    if (body_start != std::string::npos)
        req.body = raw.substr(body_start + 4);

    return req;
}

// ============================================================================
// Small socket helpers
// ============================================================================
static bool send_all(int fd, const char* data, int len) {
    int sent = 0;
    while (sent < len) {
        int n = (int)::send(fd, data + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

static bool send_all(int fd, const std::string& s) {
    return send_all(fd, s.data(), (int)s.size());
}

static void set_recv_timeout(int fd, int ms) {
#ifdef _WIN32
    DWORD t = (DWORD)ms;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&t, sizeof(t));
#else
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

static void set_no_delay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
}

// Read the full request: headers + body (per Content-Length), so JSON POST
// bodies are never truncated by a single recv.  Returns whatever we got on
// error/timeout so the handler can still respond.
HttpRequest read_request(int fd) {
    HttpRequest req;
    std::string buf;
    char tmp[16384];
    for (int round = 0; round < 64; round++) {  // cap ~1 MB
        int n = (int)::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        buf.append(tmp, n);

        size_t hdr_end = buf.find("\r\n\r\n");
        if (hdr_end == std::string::npos) continue;

        // Content-Length: <n>  (case-insensitive prefix match)
        size_t p = buf.find("Content-Length:");
        if (p == std::string::npos) p = buf.find("content-length:");
        if (p == std::string::npos) break;  // no body expected
        p += 15;                            // len("Content-Length:") == 15
        while (p < buf.size() && buf[p] == ' ') p++;
        int cl = 0;
        while (p < buf.size() && buf[p] >= '0' && buf[p] <= '9') {
            cl = cl * 10 + (buf[p] - '0');
            p++;
        }
        if ((int)buf.size() - (int)hdr_end - 4 >= cl) break;
    }
    return parse_request(buf.data(), (int)buf.size());
}

// ============================================================================
// Socket-backed HttpResponseWriter
// ============================================================================
class SocketWriter : public HttpResponseWriter {
public:
    explicit SocketWriter(int fd) : fd_(fd) {}

    ~SocketWriter() override {
        // If a handler began a stream but never ended it (exception / early
        // return), terminate the chunked body so the client doesn't hang.
        if (streaming_) send_all(fd_, "0\r\n\r\n");
    }

    void send(int status, const std::string& content_type,
              const std::string& body) override {
        std::ostringstream oss;
        oss << "HTTP/1.1 " << status << " " << reason(status) << "\r\n";
        oss << "Content-Type: " << content_type << "\r\n";
        oss << "Content-Length: " << body.size() << "\r\n";
        oss << "Access-Control-Allow-Origin: *\r\n";
        oss << "Connection: close\r\n";
        oss << "\r\n";
        oss << body;
        send_all(fd_, oss.str());
    }

    void begin_stream(int status, const std::string& content_type) override {
        std::ostringstream oss;
        oss << "HTTP/1.1 " << status << " " << reason(status) << "\r\n";
        oss << "Content-Type: " << content_type << "\r\n";
        oss << "Transfer-Encoding: chunked\r\n";
        oss << "Access-Control-Allow-Origin: *\r\n";
        oss << "Connection: close\r\n";
        oss << "\r\n";
        send_all(fd_, oss.str());
        streaming_ = true;
    }

    void send_chunk(const std::string& data) override {
        std::ostringstream oss;
        oss << std::hex << data.size() << std::dec << "\r\n";
        oss << data << "\r\n";
        send_all(fd_, oss.str());
    }

    void end_stream() override {
        if (streaming_) {
            send_all(fd_, "0\r\n\r\n");
            streaming_ = false;
        }
    }

private:
    static const char* reason(int status) {
        switch (status) {
            case 200: return "OK";
            case 400: return "Bad Request";
            case 404: return "Not Found";
            case 500: return "Internal Server Error";
            default:  return "OK";
        }
    }

    int fd_;
    bool streaming_ = false;
};

}  // namespace

// ============================================================================
// run_server
// ============================================================================
void run_server(int port, HttpHandler handler) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    int sock = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { fprintf(stderr, "socket failed\n"); return; }

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "bind failed (port %d in use?)\n", port);
        closesocket(sock);
        return;
    }
    if (listen(sock, 64) < 0) {
        fprintf(stderr, "listen failed\n");
        closesocket(sock);
        return;
    }
    printf("NanoInfer HTTP server listening on http://localhost:%d\n", port);
    fflush(stdout);

    while (true) {
        struct sockaddr_in client;
        socklen_t client_len = sizeof(client);
        int client_sock = (int)accept(sock, (struct sockaddr*)&client,
                                      &client_len);
        if (client_sock < 0) {
            // accept failed (transient) — poll briefly and retry
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Don't let a half-open connection hold the thread forever.
        set_recv_timeout(client_sock, 30000);
        set_no_delay(client_sock);   // streaming: flush each chunk promptly

        std::thread([handler, client_sock]() {
            HttpRequest req = read_request(client_sock);
            SocketWriter writer(client_sock);
            try {
                handler(req, writer);
            } catch (...) {
                fprintf(stderr, "http handler threw\n");
            }
            closesocket(client_sock);
        }).detach();
    }
    closesocket(sock);
}

}  // namespace server
}  // namespace nanoinfer
