# Minimal sandbox image for running the opencode analyzer CLI.
# Build:  docker build -t arcgym/opencode-sandbox:latest .

# ---- Stage 1: fetch the opencode binary ----
FROM python:3.12-slim AS fetcher

# Allow apt to build under rootless (single-UID) podman: don't drop to _apt (uid 42),
# which is unmappable without a subuid range.
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/00rootless

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --no-same-owner so tar doesn't try to restore archived uids under rootless build.
ENV TAR_OPTIONS=--no-same-owner
RUN curl -fsSL https://opencode.ai/install | bash \
    && mv /root/.opencode/bin/opencode /usr/local/bin/opencode \
    && chmod +x /usr/local/bin/opencode

# ---- Stage 2: minimal runtime ----
FROM python:3.12-slim

RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/00rootless

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/false opencode

COPY --from=fetcher /usr/local/bin/opencode /usr/local/bin/opencode

# Rootless single-UID: stay as root-in-userns (maps to host uid). The agent also
# drops its --user 1000:1000 flag to match (see opencode_agent.py rootless patch).
WORKDIR /workspace
ENTRYPOINT ["opencode"]
