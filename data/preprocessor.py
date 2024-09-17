"""Text preprocessing for academic papers."""
import re
from typing import Optional


class TextPreprocessor:
    """Preprocesses text for embedding generation."""

    LATEX_PATTERNS = [
        (r"\$[^$]+\$", " [MATH] "),  # Inline math
        (r"\\\[.+?\\\]", " [EQUATION] "),  # Display math
        (r"\\[a-zA-Z]+\{[^}]*\}", ""),  # LaTeX commands
        (r"\\[a-zA-Z]+", ""),  # Simple LaTeX commands
        (r"\{|\}", ""),  # Curly braces
    ]

    def __init__(
        self,
        remove_latex: bool = True,
        lowercase: bool = False,
        max_length: Optional[int] = None,
    ):
        self.remove_latex = remove_latex
        self.lowercase = lowercase
        self.max_length = max_length

    def clean_latex(self, text: str) -> str:
        """Remove LaTeX formatting from text."""
        for pattern, replacement in self.LATEX_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.DOTALL)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def remove_special_chars(self, text: str) -> str:
        """Remove special characters while preserving meaning."""
        text = re.sub(r"[^\w\s.,;:!?\-()'\"/]", " ", text)
        return text

    def process(self, text: str) -> str:
        """Apply all preprocessing steps."""
        if not text:
            return ""

        if self.remove_latex:
            text = self.clean_latex(text)

        text = self.normalize_whitespace(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)

        if self.lowercase:
            text = text.lower()

        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]

        return text

    def combine_title_abstract(self, title: str, abstract: str, sep: str = " ") -> str:
        """Combine title and abstract for embedding."""
        title = self.process(title)
        abstract = self.process(abstract)
        return f"{title}{sep}{abstract}".strip()


def preprocess_papers(
    papers: list,
    preprocessor: Optional[TextPreprocessor] = None,
) -> list[str]:
    """Preprocess a list of papers for embedding."""
    if preprocessor is None:
        preprocessor = TextPreprocessor()

    texts = []
    for paper in papers:
        text = preprocessor.combine_title_abstract(paper.title, paper.abstract)
        texts.append(text)

    return texts
