import re
from typing import List


# Consider using LangChain's text splitters for more advanced strategies later,
# but let's start with a simpler custom one.
# from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        # self.splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap,
        #     length_function=len,
        #     is_separator_regex=False,
        #     separators=["\n\n", "\n", " ", ""] # Common separators
        # )
        print(f"TextProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def clean_text(self, text: str) -> str:
        """
        Performs basic text cleaning.
        - Removes excessive whitespace.
        - (Optionally) Can add more cleaning like HTML tag removal, unicode normalization, etc.
        """
        if not text:
            return ""
        # Remove multiple newlines, replace with a single one
        text = re.sub(r'\n\s*\n', '\n', text)
        # Remove multiple spaces, replace with a single one
        text = re.sub(r'\s\s+', ' ', text)
        text = text.strip()
        return text

    def chunk_text_simple(self, text: str) -> List[str]:
        """
        A very simple character-based chunking strategy with overlap.
        This is a basic implementation. For production, consider more robust sentence-aware
        or token-aware chunking, or libraries like LangChain's text splitters.
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start_index = 0
        while start_index < len(text):
            end_index = min(start_index + self.chunk_size, len(text))
            chunks.append(text[start_index:end_index])

            # Move start_index for the next chunk
            # If we're at the end, break
            if end_index == len(text):
                break

            start_index += (self.chunk_size - self.chunk_overlap)

            # Ensure we don't create tiny overlapping segments if overlap is large
            if start_index >= end_index:  # Should not happen with typical overlap < chunk_size
                start_index = end_index  # Move to the end of the last chunk to avoid issues

        return chunks

    # Example using LangChain's splitter (if you install langchain)
    # def chunk_text_langchain(self, text: str) -> List[str]:
    #     if not text:
    #         return []
    #     return self.splitter.split_text(text)

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

    processor_large_overlap = TextProcessor(chunk_size=100, chunk_overlap=80)
    print(f"\n--- Long Text with Large Overlap ---")
    chunks_large_overlap = processor_large_overlap.process_and_chunk(cleaned_long)
    print(f"\nChunks (size={processor_large_overlap.chunk_size}, overlap={processor_large_overlap.chunk_overlap}):")
    for i, chunk in enumerate(chunks_large_overlap):
        print(f"Chunk {i + 1} (len={len(chunk)}): '{chunk}'")
