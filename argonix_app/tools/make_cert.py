"""Make a self-signed TLS certificate for `run.py --tls-cert/--tls-key` (LAN demo use).

Needs the `cryptography` package, which Argus itself does not depend on, so run it through uv:

    uv run --with cryptography python tools/make_cert.py                      # localhost + this machine's names/IPs
    uv run --with cryptography python tools/make_cert.py --name argonix.lan --ip 192.168.1.20

Writes data/tls/cert.pem and data/tls/key.pem (git-ignored; key is 0600). Browsers will warn because nobody vouches
for a self-signed certificate; accept it once per browser, or use mkcert (a locally trusted dev CA) instead:
    mkcert -install && mkcert -cert-file data/tls/cert.pem -key-file data/tls/key.pem localhost 127.0.0.1 <lan-ip>
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import os
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def local_names() -> tuple[list[str], list[str]]:
    names, ips = {"localhost", socket.gethostname()}, {"127.0.0.1", "::1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0].split("%")[0]
            if not ipaddress.ip_address(ip).is_link_local:
                ips.add(ip)
    except (OSError, ValueError):
        pass
    return sorted(names), sorted(ips)


def main() -> None:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError:
        raise SystemExit("needs `cryptography`: uv run --with cryptography python tools/make_cert.py  (or use mkcert)")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", action="append", default=[], help="extra DNS name (repeatable)")
    ap.add_argument("--ip", action="append", default=[], help="extra IP address (repeatable)")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", default=str(ROOT / "data" / "tls"))
    a = ap.parse_args()
    names, ips = local_names()
    names = sorted(set(names) | set(a.name))
    ips = sorted(set(ips) | set(a.ip))
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Argus (self-signed)")])
    now = dt.datetime.now(dt.timezone.utc)
    san = [x509.DNSName(n) for n in names] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=a.days))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    fd = os.open(out / "key.pem", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    print(f"wrote {out / 'cert.pem'} and {out / 'key.pem'} for {', '.join(names + ips)} (valid {a.days} days)")
    print(f"run:  uv run python run.py --tls-cert {out / 'cert.pem'} --tls-key {out / 'key.pem'}")


if __name__ == "__main__":
    main()
