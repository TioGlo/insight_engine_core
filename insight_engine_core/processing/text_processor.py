import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

HARD_SPLIT_THRESHOLD_CHARS = 1000  # Default, can be overridden in tests by re-init


class TextProcessor:
    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 separators: List[str] = None,
                 hard_split_threshold: int = HARD_SPLIT_THRESHOLD_CHARS):
        self.chunk_size = chunk_size  # Target for primary semantic splitting
        self.chunk_overlap = chunk_overlap
        self.hard_split_threshold = hard_split_threshold

        if separators is None:
            self.separators = ["\n\n", "\n", "\t", " ", ""]
        else:
            self.separators = separators

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators,
            keep_separator=False
        )
        # Fallback splitter's goal is to break chunks that are > hard_split_threshold
        # So its own chunk_size should be aimed at or below hard_split_threshold.
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.hard_split_threshold,  # Use hard_split_threshold as target
            chunk_overlap=min(self.chunk_overlap,
                              self.hard_split_threshold // 4 if self.hard_split_threshold > 0 else 0),
            # Smaller overlap for hard splits
            length_function=len,
            separators=[" ", ""],  # Aggressive: split by space, then by character
            keep_separator=False
        )
        print(
            f"TextProcessor: target_chunk_size={chunk_size}, overlap={chunk_overlap}, hard_threshold={self.hard_split_threshold}, fallback_chunk_size={self.fallback_splitter._chunk_size}")

    def clean_text(self, text: str) -> str:
        if not text: return ""

        # 1. Normalize various non-newline whitespace characters (tabs, etc.) to spaces.
        #    Also, collapse multiple spaces within lines to a single space here.
        text = re.sub(r'[ \t\r\f\v]+', ' ', text)

        # 2. Replace any run of 2 or more newlines (now that other whitespace between them is gone or single space)
        #    with a unique paragraph break marker.
        #    This regex looks for (newline, optional single space, newline) repeated, or just multiple newlines.
        text = re.sub(r'(\n[ ]*){2,}', '<<PARAGRAPH_BREAK>>', text)  # Handles \n\n, \n \n, \n\n\n etc.

        # 3. For any remaining single newline (that wasn't part of a paragraph break),
        #    if it's followed by spaces, remove those spaces.
        #    This handles "\n   SomeText" -> "\nSomeText".
        #    It will NOT affect "<<PARAGRAPH_BREAK>>"
        text = re.sub(r'\n[ ]+', '\n', text)  # Only spaces, not \s which includes \n

        # 4. Restore the paragraph break markers to double newlines.
        text = text.replace('<<PARAGRAPH_BREAK>>', '\n\n')

        # 5. Collapse multiple literal spaces (that might have been formed or missed)
        #    This is a final cleanup for spaces within lines.
        text = re.sub(r' {2,}', ' ', text)

        # 6. Strip leading/trailing whitespace (including newlines if any) from the entire string.
        text = text.strip()

        return text

    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []

        semantic_chunks = self.recursive_splitter.split_text(text)

        final_chunks = []
        for chunk in semantic_chunks:
            if len(chunk) > self.hard_split_threshold:
                print(
                    f"INFO: Chunk (len={len(chunk)}) exceeded hard_split_threshold ({self.hard_split_threshold}). Applying fallback split.")
                further_split_chunks = self.fallback_splitter.split_text(chunk)
                final_chunks.extend(further_split_chunks)
            else:
                final_chunks.append(chunk)
        return final_chunks

    def process_and_chunk(self, text: str) -> List[str]:
        cleaned_text = self.clean_text(text)
        chunks = self.chunk_text(cleaned_text)
        return chunks


if __name__ == '__main__':
    processor = TextProcessor(chunk_size=100, chunk_overlap=20)

    sample_text_long = """This is a longer piece of text. It has multiple sentences.\n\nWe want to see how it gets chunked.
    Each chunk should be around the specified chunk size, with some overlap.
    Extra spaces should be handled.

    Another paragraph here.\tA tabbed sentence.
    And another line in the same paragraph.
    """
    print(f"\n--- Long Text ---")
    print(f"Original:\n{sample_text_long}")
    # Test clean_text separately
    cleaned_sample = processor.clean_text(sample_text_long)
    print(f"\nCleaned for LangChain splitter:\n{cleaned_sample}")

    chunks_long = processor.process_and_chunk(sample_text_long)
    print(f"\nChunks (size={processor.chunk_size}, overlap={processor.chunk_overlap}) using LangChain:")
    for i, chunk in enumerate(chunks_long):
        print(f"Chunk {i + 1} (len={len(chunk)}): '{chunk}'")
