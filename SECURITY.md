# Security policy

Argus is a personal demo and research project, not a supported product. It handles camera credentials, video evidence and operator accounts, so security reports are still very welcome.

## Reporting a vulnerability

Please **don't** open a public issue. Use GitHub's private vulnerability reporting instead: the repository's **Security** tab → **Report a vulnerability**. Include the affected file or endpoint, the steps to reproduce, and what an attacker gains. We aim to acknowledge reports within 5 days.

## Supported versions

| Version | Supported |
|---|---|
| `main` | yes |
| anything older | no |

## Scope

In scope: the app in `argonix_app/` (web server, API, auth and sessions, camera and sensor ingestion, evidence export, the VLM and inference-node services in `tools/`) and the configs in this repository.

Out of scope: vulnerabilities in third-party dependencies (report them upstream; tell us if Argus uses the affected code), and the design manual and research documents.

## What the app already does

These are described in `argonix_app/README.md` ("Accounts & access", "Hardening", "Evidence integrity"):

- accounts with roles and permissions, idle and absolute session limits, login throttling, HTTP Basic and API tokens for scripts;
- a first-run setup code instead of a default password; `--open-demo` disables login and is for local demos only;
- a CSRF header check on state-changing requests, security headers and a CSP;
- camera passwords redacted in API responses, logs and the audit trail;
- HMAC-signed sensor webhooks with replay protection;
- a SHA-256 hash-chained audit log and hash-chained evidence export bundles.

## Known limitations

- TLS is optional. Without it, run behind a reverse proxy or keep the server on a trusted LAN or `127.0.0.1` with an SSH tunnel.
- The VLM server and remote inference nodes are meant to listen on localhost or a private network only.
- WebRTC live view has no TURN relay; media flows peer-to-peer on the LAN.
