import os
import mimetypes
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from typing import Optional
from src.demo.session_manager import NavigationSessionManager
from src.demo.replay_controller import ReplayController
from src.demo.api import DemoAPIHandler


class NavigationHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving web assets and API endpoints."""

    api_handler: Optional[DemoAPIHandler] = None
    web_dir: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.web_dir, **kwargs)

    def log_message(self, format, *args):
        """Suppress noisy access logging for high-rate polling."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path.startswith("/api/"):
            status, headers, body = self.api_handler.handle_request("GET", self.path)
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            # Default to index.html for root path
            if self.path == "/" or self.path == "":
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b""

        if self.path.startswith("/api/"):
            status, headers, body = self.api_handler.handle_request("POST", self.path, body_bytes)
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        """Handle OPTIONS preflight requests."""
        status, headers, body = self.api_handler.handle_request("OPTIONS", self.path)
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DemoServer:
    """Embedded HTTP server hosting UI dashboard and navigation telemetry API."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        checkpoint_path: Optional[str] = None,
        web_dir: Optional[str] = None,
    ):
        """Initialize demo server."""
        self.host = host
        self.port = port
        self.session = NavigationSessionManager(checkpoint_path=checkpoint_path)
        self.controller = ReplayController(self.session)
        self.api_handler = DemoAPIHandler(self.session, self.controller)

        if web_dir is None:
            # Relative to project root
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            web_dir = os.path.join(base_dir, "web")
        self.web_dir = web_dir
        os.makedirs(self.web_dir, exist_ok=True)

        # Configure handler class
        NavigationHTTPRequestHandler.api_handler = self.api_handler
        NavigationHTTPRequestHandler.web_dir = self.web_dir

        self.httpd: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    def start(self, auto_replay_loop: bool = True, background_http: bool = True):
        """Start background replay loop and bind HTTP server."""
        if auto_replay_loop:
            self.controller.start_loop()

        # Try binding to port, with fallback if busy
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                self.httpd = ThreadingHTTPServer((self.host, self.port), NavigationHTTPRequestHandler)
                break
            except OSError:
                self.port += 1

        if self.httpd is None:
            raise RuntimeError(f"Failed to bind HTTP server to {self.host}:{self.port}")

        if background_http:
            self._server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self._server_thread.start()

    def serve_forever(self):
        """Start listening for incoming HTTP requests (blocking)."""
        if self.httpd is None:
            self.start(auto_replay_loop=True, background_http=False)
        print(f"SIH-26168 Interactive Navigation Prototype running at: http://{self.host}:{self.port}")
        self.httpd.serve_forever()

    def shutdown(self):
        """Clean shutdown of server and worker threads."""
        self.controller.stop_loop()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None

    def get_url(self) -> str:
        """Return full browser URL."""
        return f"http://{self.host}:{self.port}"

