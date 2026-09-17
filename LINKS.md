# Links

The canonical home for every URL this project publishes.

Four artifacts cite each other. When they each held their own copy of these
addresses, renaming a repository or changing a Streamlit subdomain rotted the
links silently in three places at once — the same staleness problem
`src/reporting/facts.py` solves for the numbers. Everything points here instead,
so a URL changes in exactly one file.

---

## Live

| What | URL |
|---|---|
| **Report site** | https://bala-shunmugam-m.github.io/ai-data-science-workbench/ |
| **Streamlit app** | https://ai-data-science-workbench-ivbkapzndlvvyjnxhu6wk6.streamlit.app/ |

The report site is static: generated reports, model cards and fairness audits,
republished by CI on every push to `main`.

The Streamlit app is the live interactive demo — upload a CSV and watch the
governed pipeline run on it. Free tier, so it may take a few seconds to wake.

---

## Source

| What | URL |
|---|---|
| **Main repository** | https://github.com/Bala-Shunmugam-M/ai-data-science-workbench |
| **Console screenshot tour** | https://github.com/Bala-Shunmugam-M/ai-data-science-workbench-console |

The tour is presentation only — eight captures of a real run. The console's code
lives in the main repository, not there, so there is no second copy to drift.

---

## Not hosted, and why

The **Next.js console** runs locally only. Its API routes spawn the pipeline as a
child process, so a host must run Node and Python in the same container. That
rules out GitHub Pages, Vercel and Netlify outright.

Container hosts can do it — `deploy/render/Dockerfile` and `render.yaml` are
ready — but every remaining free tier (Render, Fly.io, Railway, Koyeb) requires a
payment card, and Hugging Face Spaces bills for any Space that runs Python.
Probed directly against a free account:

    docker     BLOCKED - PRO subscription required
    gradio     BLOCKED - PRO subscription required
    streamlit  BLOCKED - PRO subscription required
    static     ALLOWED

---

## Maintaining this file

If a URL changes, change it **here** and nowhere else. The other documents link
to this page rather than repeating the address. A dead link in one place is a
bug; the same dead link in four places is a maintenance problem.
