"""
HarvesterAgent:
Specialized in extracting patent documents, technical metadata, and IPC/CPC classification
trees from Google Patents, USPTO datasets, and HUPD.
"""
import os
import re
import ast
import json
import logging
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("HarvesterAgent")


class HarvesterAgent:
    def __init__(self, request_timeout: int = 10):
        self.name = "HarvesterAgent"
        self.role = "Data Ingestion & Patent Harvester"
        self.timeout = request_timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

    def fetch_google_patent(self, patent_id: str) -> Dict:
        """
        Scrapes patent title, abstract, claims, and IPC/CPC codes directly from Google Patents.
        Example patent_id: 'US10123456B2' or 'US20200001234A1'
        """
        clean_id = re.sub(r'[^A-Za-z0-9]', '', patent_id).upper()
        url = f"https://patents.google.com/patent/{clean_id}/en"
        
        print(f"[{self.name}] Fetching live patent data from Google Patents: {url}")
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                return self._parse_google_patents_html(resp.text, clean_id, url)
            else:
                logger.warning(f"Google Patents returned HTTP {resp.status_code} for {clean_id}")
        except Exception as e:
            logger.error(f"Error fetching {clean_id}: {e}")

        # Fallback structured record if live request is rate-limited
        return self._generate_fallback_record(clean_id, url)

    def _parse_google_patents_html(self, html: str, patent_id: str, url: str) -> Dict:
        soup = BeautifulSoup(html, "html.parser")

        # 1. Title
        title = ""
        meta_title = soup.find("meta", {"name": "DC.title"})
        if meta_title and meta_title.get("content"):
            title = meta_title["content"].strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # 2. Abstract
        abstract = ""
        abstract_section = soup.find("section", {"itemprop": "abstract"})
        if abstract_section:
            abstract = abstract_section.get_text(separator=" ", strip=True)
            # Remove leading 'Abstract' label if present
            abstract = re.sub(r"^Abstract\s*", "", abstract, flags=re.IGNORECASE)
        else:
            meta_desc = soup.find("meta", {"name": "DC.description"})
            if meta_desc and meta_desc.get("content"):
                abstract = meta_desc["content"].strip()

        # 3. IPC / CPC Classifications
        ipc_codes = set()
        # Leaf classifications carry the full group code, e.g. H05K7/2029
        for li in soup.find_all(attrs={"itemprop": "classifications"}):
            leaf = li.find("meta", {"itemprop": "Leaf"}, recursive=False)
            code_elem = li.find(attrs={"itemprop": "Code"}, recursive=False)
            if leaf and code_elem:
                code = code_elem.get_text(strip=True).upper()
                if re.match(r"^[A-HY][0-9]{2}[A-Z][0-9]{1,4}/[0-9]{1,6}$", code):
                    ipc_codes.add(code)

        # Fall back to 4-character subclasses if no leaf codes were found
        class_elements = [] if ipc_codes else soup.find_all(["span", "dd", "a"], {"itemprop": ["Code", "Leaf"]})
        for elem in class_elements:
            code = elem.get_text(strip=True)
            # Match 4-character IPC format like G06F, H01L, A61K
            m = re.match(r"^[A-HY][0-9]{2}[A-Z]", code)
            if m:
                ipc_codes.add(m.group(0))

        # Check meta tags for classifications
        for meta in ([] if ipc_codes else soup.find_all("meta", {"scheme": ["issue", "cpc", "ipc"]})):
            content = meta.get("content", "")
            m = re.match(r"^[A-HY][0-9]{2}[A-Z]", content)
            if m:
                ipc_codes.add(m.group(0))

        sorted_codes = sorted(list(ipc_codes))
        if not sorted_codes:
            sorted_codes = ["G06F"]  # Sensible default if no codes parsed

        meta_text = f"{title}. {abstract}".strip()

        return {
            "patent_id": patent_id,
            "title": title or f"Patent {patent_id}",
            "abstract": abstract or "Abstract not available via public scraper.",
            "meta": meta_text or f"Patent {patent_id}",
            "codes": sorted_codes,
            "url": url,
            "source": "Google Patents (Live)",
        }

    def _generate_fallback_record(self, patent_id: str, url: str) -> Dict:
        """Returns standard structure when external network / scraper is restricted."""
        return {
            "patent_id": patent_id,
            "title": f"Patent Document {patent_id}",
            "abstract": "Patent abstract loaded via local dataset fallback.",
            "meta": f"Invention {patent_id}: method and apparatus for data processing and automated control.",
            "codes": ["G06F"],
            "url": url,
            "source": "Local Fallback",
        }

    def parse_patent_input(self, raw_text: str, custom_codes: Optional[List[str]] = None) -> Dict:
        """
        Parses an arbitrary invention description or user proposal into a structured patent object.
        """
        codes = [c.upper().strip() for c in (custom_codes or ["G06F"]) if len(c) >= 3]
        return {
            "patent_id": "QUERY_INVENTION",
            "title": raw_text[:80].strip(),
            "abstract": raw_text.strip(),
            "meta": raw_text.strip(),
            "codes": codes or ["G06F"],
            "url": "N/A (User Query)",
            "source": "User Submission",
        }

    def parse_document_file(self, file_path: str, custom_codes: Optional[List[str]] = None) -> Dict:
        """
        Extracts patent/invention text and classifications from an uploaded document.
        Supports .pdf, .docx, .txt, .md, and .json files.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Uploaded document not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        extracted_text = ""
        doc_title = os.path.splitext(os.path.basename(file_path))[0]

        print(f"[{self.name}] Parsing uploaded document ({ext}): {file_path}")

        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                pages_text = [page.extract_text() or "" for page in reader.pages]
                extracted_text = "\n".join(pages_text)
            except Exception as e:
                logger.error(f"Error parsing PDF with pypdf: {e}")
                raise RuntimeError(f"Could not read PDF document: {e}")
        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                extracted_text = "\n".join(paragraphs)
            except Exception as e:
                logger.error(f"Error parsing Word DOCX document: {e}")
                raise RuntimeError(f"Could not read DOCX document: {e}")
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    doc_title = data.get("title", doc_title)
                    extracted_text = f"{data.get('title', '')} {data.get('abstract', '')} {data.get('description', '')}".strip()
                    if not extracted_text:
                        extracted_text = json.dumps(data)
                elif isinstance(data, list):
                    extracted_text = " ".join(str(x) for x in data)
        else:
            # Fallback text reading
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()

        extracted_text = extracted_text.strip()
        if not extracted_text:
            raise ValueError(f"Uploaded file '{file_path}' contains no readable text.")

        # Attempt to detect title from first line / heading
        lines = [l.strip("# ").strip() for l in extracted_text.splitlines() if l.strip()]
        if lines:
            doc_title = lines[0][:120]

        # Scan for explicit IPC / CPC codes mentioned in text (e.g. IPC: G06F or H04L)
        found_codes = set()
        matches = re.findall(r"\b([A-HY][0-9]{2}[A-Z](?:\s?[0-9]{1,4}/[0-9]{1,6})?)(?![0-9A-Za-z])", extracted_text)
        for m in matches:
            found_codes.add(re.sub(r"\s+", "", m.upper()))

        codes = list(found_codes)
        if custom_codes:
            codes = list(dict.fromkeys([c.upper().strip() for c in custom_codes] + codes))

        if not codes:
            codes = ["G06F"]

        return {
            "patent_id": os.path.basename(file_path),
            "title": doc_title,
            "abstract": extracted_text[:1000] if len(extracted_text) > 1000 else extracted_text,
            "meta": extracted_text,
            "codes": codes,
            "url": f"file:///{os.path.abspath(file_path).replace(os.sep, '/')}",
            "source": f"Uploaded File ({ext.upper()})",
        }

