"""HTTPS with exact origin allowlists, pinned public DNS and bounded bodies.

No proxy environment, cookies, automatic redirects or automatic decompression.
TLS verifies the original hostname while the socket connects to a vetted IP.
"""

import http.client
import ipaddress
import json
import socket
import ssl
from dataclasses import dataclass
from urllib.parse import urlsplit


class AcquisitionError(ValueError):
    pass


def origin(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.port not in (None, 443)
            or any(ord(c) < 33 or ord(c) == 127 for c in url)
            or "\\" in url
        ):
            raise ValueError()
        return f"https://{parsed.hostname.lower()}"
    except ValueError:
        raise AcquisitionError("Invalid HTTPS origin") from None


def validate_url(url: str, allowed_origins: set[str]) -> tuple[str, list[str]]:
    if origin(url) not in allowed_origins:
        raise AcquisitionError("Origin is not registered")
    host = urlsplit(url).hostname or ""
    try:
        addresses = sorted({str(row[4][0]) for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
        if not addresses or any(not ipaddress.ip_address(addr).is_global for addr in addresses):
            raise AcquisitionError("Non-public network destination")
    except OSError:
        raise AcquisitionError("Source DNS unavailable") from None
    return host, addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str):
        self.tls_context = ssl.create_default_context()
        super().__init__(host, timeout=10, context=self.tls_context)
        self.address = address

    def connect(self) -> None:
        raw = socket.create_connection((self.address, 443), self.timeout)
        try:
            self.sock = self.tls_context.wrap_socket(raw, server_hostname=self.host)
        except Exception:
            raw.close()
            raise


@dataclass(frozen=True)
class Acquired:
    body: bytes
    mime: str


class SafeHTTP:
    def __init__(self, allowed_origins: set[str], max_bytes: int = 2_000_000):
        self.allowed_origins = allowed_origins
        self.max_bytes = max_bytes

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict | None = None,
        body: bytes | None = None,
        mime_types: tuple[str, ...] = ("application/json", "application/fhir+json"),
    ) -> Acquired:
        host, addresses = validate_url(url, self.allowed_origins)
        parsed = urlsplit(url)
        path = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
        connection = _PinnedHTTPSConnection(host, addresses[0])
        try:
            connection.request(
                method,
                path,
                body=body,
                headers={"Accept": ", ".join(mime_types), "Accept-Encoding": "identity", **(headers or {})},
            )
            response = connection.getresponse()
            # Reject all redirects, including redirects on allowlisted hosts.
            if response.status != 200:
                raise AcquisitionError(f"Upstream response rejected ({response.status})")
            mime = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if mime not in mime_types or response.getheader("Content-Encoding", "identity") != "identity":
                raise AcquisitionError("Unsupported content type or encoding")
            length = response.getheader("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > self.max_bytes):
                raise AcquisitionError("Upstream content exceeds limit")
            data = response.read(self.max_bytes + 1)
            if len(data) > self.max_bytes:
                raise AcquisitionError("Upstream content exceeds limit")
            if length is not None and len(data) != int(length):
                raise AcquisitionError("Upstream content length mismatch")
            return Acquired(data, mime)
        except (OSError, http.client.HTTPException):
            raise AcquisitionError("Upstream request failed") from None
        finally:
            connection.close()

    def json(self, url: str, **kwargs) -> dict:
        try:
            value = json.loads(self.request(url, **kwargs).body)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (ValueError, RecursionError):
            raise AcquisitionError("Invalid upstream JSON") from None
