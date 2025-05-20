import re
from typing import List


# Consider using LangChain's text splitters for more advanced strategies later,
# but let's start with a simpler custom one.
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the TextProcessor.
        Args:
            chunk_size: The target size of each chunk (in characters).
            chunk_overlap: The number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # For more sophisticated splitting, you might use LangChain's splitters:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""] # Common separators
        )
        print(f"TextProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        # 1. Normalize various non-newline whitespace characters (tabs, etc.) to spaces.
        text = re.sub(r'[ \t\r\f\v]+', ' ', text)

        # 2. Replace any run of 2 or more newlines (possibly with spaces between them)
        #    with a unique paragraph break marker. This helps preserve intended paragraphs.
        #    (\n\s*){2,} matches two or more occurrences of (newline followed by optional spaces).
        text = re.sub(r'(\n\s*){2,}', '<<PARAGRAPH_BREAK>>', text)

        # 3. For any remaining single newline, remove leading spaces/tabs that immediately follow it.
        #    This handles cases like "\n   SomeText" -> "\nSomeText".
        text = re.sub(r'\n\s+', '\n', text)

        # 4. Restore the paragraph break markers to double newlines.
        text = text.replace('<<PARAGRAPH_BREAK>>', '\n\n')

        # 5. Collapse multiple spaces (that are not part of newlines) into a single space.
        #    This cleans up spaces within lines that might have been created or were already there.
        text = re.sub(r' {2,}', ' ', text)  # Note: ' ' specifically, not '\s'

        # 6. Strip leading/trailing whitespace (including newlines if any) from the entire string.
        text = text.strip()

        return text

    def chunk_text_simple(self, text: str) -> List[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start_index = 0
        print(
            f"CHUNK_DEBUG: Initial: text_len={len(text)}, chunk_size={self.chunk_size}, overlap={self.chunk_overlap}, step={self.chunk_size - self.chunk_overlap}")  # DEBUG
        while start_index < len(text):
            end_index = min(start_index + self.chunk_size, len(text))
            current_chunk = text[start_index:end_index]  # Store before append for debug
            chunks.append(current_chunk)
            print(f"CHUNK_DEBUG: start={start_index}, end={end_index}, chunk='{current_chunk}'")  # DEBUG

            if end_index == len(text):
                print("CHUNK_DEBUG: Reached end of text, breaking.")  # DEBUG
                break

            start_index += (self.chunk_size - self.chunk_overlap)
            print(f"CHUNK_DEBUG: Next start_index={start_index}")  # DEBUG
        return chunks

    # Example using LangChain's splitter (if you install langchain)
    def chunk_text_langchain(self, text: str) -> List[str]:
        if not text:
            return []
        return self.splitter.split_text(text)

    def process_and_chunk(self, text: str) -> List[str]:
        """Cleans and then chunks the text."""
        cleaned_text = self.clean_text(text)
        # chunks = self.chunk_text_langchain(cleaned_text) # If using LangChain
        chunks = self.chunk_text_simple(cleaned_text)  # Using simple chunker
        return chunks


if __name__ == '__main__':
    processor = TextProcessor(chunk_size=100, chunk_overlap=20)

    sample_text_short = "This is a short sentence."
    print(f"\n--- Short Text ---")
    print(f"Original: '{sample_text_short}'")
    cleaned_short = processor.clean_text(sample_text_short)
    print(f"Cleaned: '{cleaned_short}'")
    chunks_short = processor.process_and_chunk(sample_text_short)
    print(f"Chunks: {chunks_short}")

    sample_text_long = """This is a longer piece of text. It has multiple sentences.
    We want to see how it gets chunked into smaller pieces.
    Each chunk should be around the specified chunk size, with some overlap.
    Newlines and extra spaces   should be handled.

    Another paragraph here to make it longer and test paragraph splitting if using advanced splitters.
    The quick brown fox jumps over the lazy dog. This is just to add more content.
    """
    print(f"\n--- Long Text ---")
    # print(f"Original:\n{sample_text_long}")
    cleaned_long = processor.clean_text(sample_text_long)
    print(f"Cleaned:\n{cleaned_long}")
    chunks_long = processor.process_and_chunk(cleaned_long)
    print(f"\nChunks (size={processor.chunk_size}, overlap={processor.chunk_overlap}):")
    for i, chunk in enumerate(chunks_long):
        print(f"Chunk {i + 1} (len={len(chunk)}): '{chunk}'")

    lc_chunks = processor.chunk_text_langchain(sample_text_long)
    print(f"\n--- Langchain Chunks ---")
    print(lc_chunks)

    processor_large_overlap = TextProcessor(chunk_size=100, chunk_overlap=40)
    print(f"\n--- Long Text with Large Overlap ---")
    chunks_large_overlap = processor_large_overlap.process_and_chunk(cleaned_long)
    print(f"\nChunks (size={processor_large_overlap.chunk_size}, overlap={processor_large_overlap.chunk_overlap}):")
    for i, chunk in enumerate(chunks_large_overlap):
        print(f"Chunk {i + 1} (len={len(chunk)}): '{chunk}'")
