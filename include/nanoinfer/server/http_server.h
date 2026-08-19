#pragma once
/// Concurrent + streaming HTTP server for NanoInfer inference.
///
/// The accept loop runs on the calling thread and spawns one detached
/// std::thread per connection.  Each connection thread reads the request,
/// invokes the handler, and closes the socket.
///
/// Streaming: a handler calls begin_stream() once, then send_chunk() one or
/// more times (each chunk is one HTTP chunked-encoding block, e.g. one SSE
/// line), then end_stream() which writes the terminating "0\r\n\r\n".

#include <functional>
#include <string>

namespace nanoinfer {
namespace server {

struct HttpRequest {
    std::string method;   // "POST"
    std::string path;     // "/v1/chat/completions"
    std::string body;     // JSON payload
};

/// Abstract response writer.  A concrete implementation writes to a socket.
class HttpResponseWriter {
public:
    virtual ~HttpResponseWriter() = default;

    /// One-shot full response (Content-Length body).
    virtual void send(int status, const std::string& content_type,
                      const std::string& body) = 0;

    /// Start a chunked (streaming) response.
    virtual void begin_stream(int status, const std::string& content_type) = 0;

    /// Write one chunk (e.g. one SSE line).  Only valid after begin_stream().
    virtual void send_chunk(const std::string& data) = 0;

    /// Terminate the chunked body.
    virtual void end_stream() = 0;
};

using HttpHandler = std::function<void(const HttpRequest&, HttpResponseWriter&)>;

/// Start a blocking HTTP server on the given port.
/// Concurrent: one detached thread per connection.  Streaming supported via the
/// writer.  Returns on error; on success it blocks in the accept loop forever.
void run_server(int port, HttpHandler handler);

}  // namespace server
}  // namespace nanoinfer
