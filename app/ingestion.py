import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import re
import requests

def load_documents():

    urls = [
        # ── existing 8 ──────────────────────────────────────────────────────
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/index.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/first-steps.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/dependencies/index.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/path-params.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/query-params.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/body.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/routing.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/middleware.md",
        # ── NEW 7 ────────────────────────────────────────────────────────────
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/handling-errors.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/response-model.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/security/index.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/background-tasks.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/bigger-applications.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/testing.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/sql-databases.md",
    ]

    docs = []

    for url in urls:

        response = requests.get(url)
        text = response.text

        sections = re.split(r"\n## ", text)

        for section in sections:

            lines = section.split("\n", 1)

            title = lines[0].strip()
            title = re.sub(r"\*\*(.*?)\*\*", r"\1", title)
            title = re.sub(r"\{[^}]+\}", "", title)
            title = title.lstrip("#").strip()

            body = lines[1] if len(lines) > 1 else ""

            docs.append({
                "id": f"fastapi_{title.lower().replace(' ', '_')}",
                "framework": "fastapi",
                "version": "0.110",
                "source_url": url,
                "section_title": title,
                "text": body,
            })

    return docs

    