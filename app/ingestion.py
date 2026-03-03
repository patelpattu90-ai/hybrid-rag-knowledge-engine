import requests
from bs4 import BeautifulSoup
from typing import List, Dict


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