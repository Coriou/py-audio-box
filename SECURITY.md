# Security

Report sensitive vulnerabilities privately to the existing repository maintainer,
[Coriou](https://github.com/Coriou), using a private contact channel they make
available on their profile. If none is available, open a minimal issue at
<https://github.com/Coriou/py-audio-box> asking for a private reporting channel;
do not post exploit details, credentials, recordings or private logs publicly.
No dedicated security email or response deadline is advertised.

Include the affected revision, deployment type, impact and a minimal synthetic
reproduction. Remove tokens, personal paths and voice/audio data from evidence.

Treat the local Docker daemon and its operator as trusted: daemon access is a
host-level privilege boundary. Keep credentials private, use the installed
credential/scoping procedures, and revoke compromised tokens before replacing
them. Do not expose an installation beyond its documented network boundary.
Authentication and principal scopes do not replace host isolation.

Retain bounded upload validation, path/no-follow checks and ownership checks when
changing ingestion or artifact handling. Use synthetic audio in reports and tests;
reference voices, model access credentials and generated private audio belong in
operator-controlled storage, not repository issues or commits.
