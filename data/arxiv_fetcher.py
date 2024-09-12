"""ArXiv API fetcher for physics papers."""
import time
import re
from typing import Optional
from dataclasses import dataclass, field

import requests
import feedparser
import pandas as pd
from tqdm import tqdm


@dataclass
class Paper:
    """Represents an ArXiv paper."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    pdf_url: str = ""

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": "|".join(self.authors),
            "categories": "|".join(self.categories),
            "published": self.published,
            "updated": self.updated,
            "pdf_url": self.pdf_url,
        }


@dataclass
class ArxivFetcher:
    """Fetches papers from ArXiv API."""
    category: str = "physics.plasm-ph"
    base_url: str = "http://export.arxiv.org/api/query"
    batch_size: int = 100
    delay_seconds: float = 3.0
    max_results: int = 10000
    date_from: Optional[str] = None  # Format: YYYYMMDD
    date_to: Optional[str] = None    # Format: YYYYMMDD
    papers: list[Paper] = field(default_factory=list)

    def _extract_arxiv_id(self, entry_id: str) -> str:
        """Extract clean ArXiv ID from entry URL."""
        match = re.search(r"(\d+\.\d+)(v\d+)?$", entry_id)
        if match:
            return match.group(1)
        return entry_id.split("/")[-1]

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _parse_entry(self, entry: dict) -> Paper:
        """Parse a single ArXiv API entry."""
        arxiv_id = self._extract_arxiv_id(entry.get("id", ""))
        title = self._clean_text(entry.get("title", ""))
        abstract = self._clean_text(entry.get("summary", ""))
        authors = [a.get("name", "") for a in entry.get("authors", [])]
        categories = [t.get("term", "") for t in entry.get("tags", [])]
        published = entry.get("published", "")
        updated = entry.get("updated", "")

        pdf_url = ""
        for link in entry.get("links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break

        return Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=categories,
            published=published,
            updated=updated,
            pdf_url=pdf_url,
        )

    def fetch_batch(self, start: int) -> list[Paper]:
        """Fetch a batch of papers starting at index."""
        # Build search query with optional date range
        query = f"cat:{self.category}"
        if self.date_from and self.date_to:
            query += f" AND submittedDate:[{self.date_from} TO {self.date_to}]"
        elif self.date_from:
            query += f" AND submittedDate:[{self.date_from} TO *]"
        elif self.date_to:
            query += f" AND submittedDate:[* TO {self.date_to}]"

        params = {
            "search_query": query,
            "start": start,
            "max_results": self.batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            return [self._parse_entry(e) for e in feed.entries]
        except requests.RequestException as e:
            print(f"Error fetching batch at {start}: {e}")
            return []

    def fetch_all(self, progress: bool = True) -> list[Paper]:
        """Fetch all papers up to max_results."""
        self.papers = []
        total_batches = (self.max_results + self.batch_size - 1) // self.batch_size

        iterator = range(0, self.max_results, self.batch_size)
        if progress:
            iterator = tqdm(iterator, total=total_batches, desc="Fetching papers")

        for start in iterator:
            batch = self.fetch_batch(start)
            if not batch:
                print(f"Empty batch at {start}, stopping...")
                break
            self.papers.extend(batch)

            if len(batch) < self.batch_size:
                break

            time.sleep(self.delay_seconds)

        print(f"Fetched {len(self.papers)} papers")
        return self.papers

    def to_dataframe(self) -> pd.DataFrame:
        """Convert papers to DataFrame."""
        if not self.papers:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.papers])

    def save(self, path: str) -> None:
        """Save papers to parquet file."""
        df = self.to_dataframe()
        df.to_parquet(path, index=False)
        print(f"Saved {len(df)} papers to {path}")

    @classmethod
    def load(cls, path: str) -> "ArxivFetcher":
        """Load papers from parquet file."""
        df = pd.read_parquet(path)
        fetcher = cls()
        fetcher.papers = [
            Paper(
                arxiv_id=row["arxiv_id"],
                title=row["title"],
                abstract=row["abstract"],
                authors=row["authors"].split("|") if row["authors"] else [],
                categories=row["categories"].split("|") if row["categories"] else [],
                published=row["published"],
                updated=row["updated"],
                pdf_url=row.get("pdf_url", ""),
            )
            for _, row in df.iterrows()
        ]
        return fetcher
