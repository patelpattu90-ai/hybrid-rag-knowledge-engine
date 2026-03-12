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
       
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/handling-errors.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/response-model.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/security/index.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/background-tasks.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/bigger-applications.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/testing.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/sql-databases.md",

        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/request-files.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/request-forms.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/cookie-params.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/header-params.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/path-operation-configuration.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/extra-data-types.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/schema-extra-example.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/response-status-code.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/metadata.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/static-files.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/cors.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/encoder.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/body-updates.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/dependencies/classes-as-dependencies.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/dependencies/sub-dependencies.md","https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/request-files.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/request-forms.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/cookie-params.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/header-params.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/path-operation-configuration.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/extra-data-types.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/schema-extra-example.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/response-status-code.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/metadata.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/static-files.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/cors.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/encoder.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/body-updates.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/dependencies/classes-as-dependencies.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/dependencies/sub-dependencies.md",
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

    