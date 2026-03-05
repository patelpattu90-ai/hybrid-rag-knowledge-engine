import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import re
import requests

def load_documents():

    urls = [
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/index.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/first-steps.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/dependencies/index.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/path-params.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/query-params.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/body.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/routing.md",
        "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/middleware.md",
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

    for section in sections:

        lines = section.split("\n", 1)

        title = lines[0].strip()
        title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)   # **text** → text
        title = re.sub(r'\{[^}]+\}', '', title)            # {#anchor} → ""
        title = title.lstrip('#').strip()                  # # Heading → Heading

        body = lines[1] if len(lines) > 1 else ""

        docs.append(
            {
                "id": "fastapi_dependencies",
                "framework": "fastapi",
                "version": "0.110",
                "source_url": url,
                "section_title": title,
                "text": body,
            }
        )

    return docs

def fetch_and_parse(url: str) -> List[Dict]:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    sections = []
    current_section = "Introduction"
    content_buffer = []

    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):

        if tag.name in ["h1", "h2", "h3"]:
            if content_buffer:
                sections.append({
                    "section_title": current_section,
                    "text": " ".join(content_buffer)
                })
                content_buffer = []

            current_section = tag.get_text(strip=True)

        else:
            text = tag.get_text(strip=True)
            if text:
                content_buffer.append(text)

    if content_buffer:
        sections.append({
            "section_title": current_section,
            "text": " ".join(content_buffer)
        })

    return sections


def ingest_urls(urls: List[str], framework: str, version: str):

    all_documents = []

    for url in urls:
        sections = fetch_and_parse(url)

        for section in sections:
            all_documents.append({
                "framework": framework,
                "version": version,
                "source_url": url,
                "section_title": section["section_title"],
                "text": section["text"]
            })

    return all_documents