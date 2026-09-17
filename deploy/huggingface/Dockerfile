# Hugging Face Space image for the AI Data Science Workbench.
#
# WHY DOCKER RATHER THAN THE STREAMLIT SDK
# ----------------------------------------
# The Streamlit SDK runs `streamlit run <app_file>` and offers no hook before
# that. This Space needs one: the repository publishes the pipeline's CODE but
# not its OUTPUT - no datasets, no model pickles - so a visitor would land on
# empty dashboards and a retention simulator with no champion to load.
#
# Docker lets the demo projects be generated at BUILD time by running the real
# pipeline on the committed sample data. The result is baked into the image, so
# the Space starts instantly and nobody waits for a model to train.

FROM python:3.13-slim

# Hugging Face Spaces runs the container as uid 1000. Creating that user here
# means the pipeline can write its workspaces at runtime; running as root and
# letting HF drop privileges later leaves every generated file unwritable.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies first, so editing application code does not invalidate the
# (slow) pip layer.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user

# Generate the demo projects with the real pipeline. If this fails the image
# must not build: a Space whose dashboards are empty looks like broken software
# rather than a missing dataset.
RUN python deploy/huggingface/bootstrap.py

EXPOSE 7860

# --server.address 0.0.0.0 is required; Streamlit binds localhost by default
# and the Space would be unreachable while appearing to have started.
# CORS/XSRF are disabled because the Space is served through HF's proxy, which
# otherwise trips Streamlit's origin check on file upload.
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
