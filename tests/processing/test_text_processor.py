import pytest
from insight_engine_core.processing.text_processor import TextProcessor


class TestTextProcessor:
    def test_clean_text_empty(self):
        processor = TextProcessor()
        assert processor.clean_text("") == ""

    def test_clean_text_no_change(self):
        processor = TextProcessor()
        text = "This is clean text."
        assert processor.clean_text(text) == text

    def test_clean_text_extra_spaces(self):
        processor = TextProcessor()
        text = "This   has  extra   spaces."
        expected = "This has extra spaces."
        assert processor.clean_text(text) == expected

    def test_clean_text_newlines(self):
        processor = TextProcessor()
        text = "Line one.\n\n\nLine two.\n Line three."
        # This is the new expected output with the refined clean_text
        expected_output = "Line one.\n\nLine two.\nLine three."
        assert processor.clean_text(text) == expected_output

    # Add a new test for just multiple spaces
    def test_clean_text_multiple_spaces(self):
        processor = TextProcessor()
        text = "This   has    many     spaces."
        expected_output = "This has many spaces."
        assert processor.clean_text(text) == expected_output

    # And a test for newlines followed by spaces
    def test_clean_text_newline_space_combo(self):
        processor = TextProcessor()
        text = "Hello\n   World\n\tAgain"
        expected_output = "Hello\nWorld\nAgain"
        assert processor.clean_text(text) == expected_output

    def test_clean_text_leading_trailing_whitespace(self):
        processor = TextProcessor()
        text = "  \n  text with whitespace  \n  "
        expected = "text with whitespace"  # After multiple newline and space reduction, then strip
        assert processor.clean_text(text) == expected

    def test_chunk_text_simple_shorter_than_chunk_size(self):
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        text = "This text is shorter than the chunk size."
        chunks = processor.chunk_text_simple(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_simple_empty_text(self):
        processor = TextProcessor()
        assert processor.chunk_text_simple("") == []

    def test_chunk_text_simple_exact_chunk_size_no_overlap_needed(self):
        processor = TextProcessor(chunk_size=10, chunk_overlap=2)
        text = "0123456789"
        chunks = processor.chunk_text_simple(text)
        assert len(chunks) == 1
        assert chunks[0] == "0123456789"

    def test_chunk_text_simple_multiple_chunks_no_overlap(self):
        processor = TextProcessor(chunk_size=10, chunk_overlap=0)
        text = "0123456789abcdefghij"
        chunks = processor.chunk_text_simple(text)
        assert len(chunks) == 2
        assert chunks[0] == "0123456789"
        assert chunks[1] == "abcdefghij"

    def test_chunk_text_simple_multiple_chunks_with_overlap(self):
        processor = TextProcessor(chunk_size=10, chunk_overlap=3)
        text = "0123456789abcdefghijklmno"  # len 25
        chunks = processor.process_and_chunk(text)  # Use process_and_chunk to also test clean_text path

        # Or if testing chunk_text_simple directly:
        # cleaned_text = processor.clean_text(text) # Assuming clean_text doesn't change this specific string
        # chunks = processor.chunk_text_simple(cleaned_text)

        assert len(chunks) == 4
        assert chunks[0] == "0123456789"
        assert chunks[1] == "789abcdefg"
        assert chunks[2] == "efghijklmn"  # Corrected expected value
        assert chunks[3] == "lmno"  # Corrected expected value

    def test_process_and_chunk_calls_clean_and_chunk(self, mocker):
        processor = TextProcessor()
        mock_clean = mocker.patch.object(processor, 'clean_text', return_value="cleaned")
        mock_chunk = mocker.patch.object(processor, 'chunk_text_simple', return_value=["chunked"])

        result = processor.process_and_chunk("raw text")

        mock_clean.assert_called_once_with("raw text")
        mock_chunk.assert_called_once_with("cleaned")
        assert result == ["chunked"]
