# Render image for the Next.js review console.
#
# WHY NODE **AND** PYTHON
# -----------------------
# The console is not a static site. Its API routes spawn the repository's own
# pipeline as a child process (`webapi/bridge.py`), so the container needs both
# runtimes in the same filesystem. That single fact rules out GitHub Pages,
# Vercel and Netlify, and is why this is a Dockerfile rather than a build
# command.
#
# WHY THE DEMOS ARE BUILT HERE
# ----------------------------
# The repository publishes the pipeline's code but not its output: datasets are
# not ours to redistribute and a .joblib is a pickle that executes on load. A
# clone therefore has empty dashboards. Seeding at BUILD time means a visitor
# never waits for a model to train, which matters on a free tier that sleeps.

FROM node:22-bookworm-slim

# Debian ships python3 but no `python`. Rather than symlinking - which quietly
# changes what `python` means for everything in the image - point the console
# at python3 through the env var its own code already reads
# (webapp/lib/python.ts: process.env.WORKBENCH_PYTHON ?? 'python').
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV WORKBENCH_PYTHON=python3 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /app

# --- Python dependencies -------------------------------------------------
# Copied alone so editing application code does not invalidate this layer.
# --break-system-packages is required on Debian bookworm (PEP 668): the image
# has no other Python consumer, so a venv would add indirection for no gain.
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# --- Node dependencies ---------------------------------------------------
# The dev dependencies are needed: TypeScript and Tailwind run at build time.
COPY webapp/package.json webapp/package-lock.json ./webapp/
RUN cd webapp && npm ci --include=dev

# --- Application ---------------------------------------------------------
COPY . .

RUN cd webapp && npm run build

# Build the demo projects with the real pipeline. Failing the build here is
# deliberate: a console whose dashboards are empty looks like broken software
# rather than a missing dataset.
RUN python3 -m src.automl.demo_seed

# Render supplies $PORT and expects the process to bind 0.0.0.0. Next binds
# localhost by default, which would leave the service unreachable while
# appearing to have started successfully.
ENV NODE_ENV=production
EXPOSE 3000
CMD ["sh", "-c", "cd webapp && npx next start -H 0.0.0.0 -p ${PORT:-3000}"]
