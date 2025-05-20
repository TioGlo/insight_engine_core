import pytest
from insight_engine_core.processing.text_processor import TextProcessor, HARD_SPLIT_THRESHOLD_CHARS


class TestTextProcessorLangChain: # New test class or adapt existing
    # tests/processing/test_text_processor.py
    def test_clean_text_for_langchain(self):  # Or its current name
        processor = TextProcessor()
        text_input = "Line one.\n\n\nLine two.\n   Line three.\tMore here."

        expected_output = "Line one.\n\nLine two.\nLine three. More here."  # Your current expected

        actual_output = processor.clean_text(text_input)

        print(f"\nDEBUG: Expected repr: {repr(expected_output)}")
        print(f"DEBUG: Actual   repr: {repr(actual_output)}")
        print(f"DEBUG: Expected len: {len(expected_output)}")
        print(f"DEBUG: Actual   len: {len(actual_output)}")

        # For character-by-character comparison if lengths are same:
        if len(expected_output) == len(actual_output):
            for i in range(len(expected_output)):
                if expected_output[i] != actual_output[i]:
                    print(
                        f"DEBUG: Difference at index {i}: Expected char '{expected_output[i]}' (ord={ord(expected_output[i])}), Actual char '{actual_output[i]}' (ord={ord(actual_output[i])})")
                    break  # Stop at first difference

        assert actual_output == expected_output

    def test_chunk_text_with_langchain_empty(self):
        processor = TextProcessor()
        assert processor.chunk_text("") == []

    def test_chunk_text_with_langchain_small_text(self):
        processor = TextProcessor(chunk_size=100, chunk_overlap=0)
        text = "This is a small text."
        cleaned = processor.clean_text(text)
        chunks = processor.chunk_text(cleaned)
        assert len(chunks) == 1
        assert chunks[0] == "This is a small text."

    def test_splits_by_paragraph_first_if_chunk_size_allows(self):
        processor = TextProcessor(chunk_size=100, chunk_overlap=10)
        text = "This is the first paragraph.\nIt has a few lines.\n\nThis is the second paragraph.\n\nThis is a third, very short."
        cleaned_text = processor.clean_text(text)

        chunks = processor.chunk_text(cleaned_text)
        print(f"Paragraph Chunks: {chunks}")

        # Check that the content is present and key paragraph breaks are somewhat honored
        # The exact number of chunks can vary based on splitter's internal logic and chunk_size.
        assert len(chunks) >= 1  # Should produce at least one chunk

        full_text_no_double_newline = cleaned_text.replace("\n\n", " ")  # Rough way to check content
        joined_chunks_no_double_newline = " ".join(chunks).replace("\n\n", " ")

        # A loose check that most content is there. This is hard with overlap.
        # For this test, let's check if the distinct paragraph contents are represented.
        assert "This is the first paragraph.\nIt has a few lines." in chunks[0]
        # The second paragraph might be merged with the first if total length < chunk_size
        # Or it might be its own chunk.
        # The third paragraph should be its own chunk or the end of the last chunk.

        # Based on actual output:
        # Chunk 0: "This is the first paragraph.\nIt has a few lines.\n\nThis is the second paragraph."
        # Chunk 1: "This is a third, very short."
        assert len(chunks) == 2  # Based on observed behavior for this chunk_size
        assert chunks[0] == "This is the first paragraph.\nIt has a few lines.\n\nThis is the second paragraph."
        assert chunks[1] == "This is a third, very short."

    def test_normal_chunking_below_hard_threshold(self):
        # Uses default HARD_SPLIT_THRESHOLD_CHARS (e.g., 1000)
        # Chunk size is 100, so chunks should be around this size from recursive_splitter
        processor = TextProcessor(chunk_size=100, chunk_overlap=10)
        text = "Short sentence one. Short sentence two.\n\nAnother paragraph here that is not too long."
        # This text should be chunked by recursive_splitter and not hit the hard threshold.

        chunks = processor.process_and_chunk(text)

        assert len(chunks) > 0
        for chunk in chunks:
            # RecursiveCharacterTextSplitter might produce slightly larger chunks
            # if it can't find a separator exactly at chunk_size.
            # We expect it to be reasonably close to chunk_size, and definitely under HARD_SPLIT_THRESHOLD_CHARS.
            assert len(
                chunk) <= processor.chunk_size + processor.chunk_overlap + 50  # Allow some leeway for word boundaries
            assert len(chunk) <= HARD_SPLIT_THRESHOLD_CHARS

            # Check if it split by paragraph
        if "\n\n" in processor.clean_text(text):  # Only if paragraphs actually exist after cleaning
            assert "Short sentence one. Short sentence two." in chunks[0]  # Or similar based on actual split
            assert "Another paragraph here that is not too long." in chunks[-1]

    def test_chunk_exceeds_hard_threshold_triggers_fallback(self, capsys):
        # Set a small hard_split_threshold for testing purposes
        test_hard_threshold = 50
        # Target chunk_size for recursive_splitter is larger than the hard_split_threshold
        # to ensure it produces a chunk that needs fallback.
        processor = TextProcessor(chunk_size=100, chunk_overlap=10, hard_split_threshold=test_hard_threshold)

        # A single long "word" or sentence with no primary separators, longer than test_hard_threshold
        # but would be a single chunk from recursive_splitter if hard_split_threshold was high.
        long_indivisible_unit = "ThisIsAVeryLongStringWithNoSpacesOrNewlinesThatWillExceedTheHardThreshold"  # len=70
        assert len(long_indivisible_unit) > test_hard_threshold
        assert len(long_indivisible_unit) < processor.chunk_size  # So recursive_splitter makes it one chunk

        chunks = processor.process_and_chunk(long_indivisible_unit)
        captured = capsys.readouterr()  # Capture print statements

        assert f"INFO: Chunk (len={len(long_indivisible_unit)}) exceeded hard_split_threshold ({test_hard_threshold})" in captured.out

        # Now check if the fallback_splitter (which uses chunk_size=100 but splits by " " and "")
        # broke it down further. Since long_indivisible_unit has no spaces, it should be char-split
        # by the fallback if its target chunk_size is smaller than len(long_indivisible_unit).
        # The fallback_splitter also uses self.chunk_size (100), so it will also return it as one chunk
        # UNLESS we make the fallback_splitter's chunk_size smaller for the test.

        # Let's refine this test: the fallback_splitter should have a smaller chunk_size
        # or the test_hard_threshold should be such that the fallback *does* split.

        # Re-test with fallback_splitter having a smaller effective chunk_size for this test
        # We can't easily change fallback_splitter's params per test without re-init or mocking.
        # Instead, let's make the long_indivisible_unit very long, and the processor's main chunk_size
        # also small, but the hard_split_threshold even smaller.

        processor_fallback_test = TextProcessor(
            chunk_size=30,  # Target for both splitters
            chunk_overlap=5,
            hard_split_threshold=20  # Force fallback
        )
        very_long_word = "SupercalifragilisticexpialidociousAndThenSomeMoreText"  # len > 30

        chunks_fb = processor_fallback_test.process_and_chunk(very_long_word)
        captured_fb = capsys.readouterr()  # Clear and capture again

        # recursive_splitter (chunk_size 30) will produce one chunk: the whole word (len > 30)
        # This chunk (len > 30) will exceed hard_split_threshold (20)
        assert f"INFO: Chunk (len={len(very_long_word)}) exceeded hard_split_threshold (20)" in captured_fb.out

        # The fallback_splitter (also chunk_size 30, but with separators " " and "")
        # should now split this long word by character because it has no spaces.
        # Each resulting chunk should be <= 30.
        assert len(chunks_fb) > 1  # It must have been split further
        for chunk in chunks_fb:
            assert len(chunk) <= 30  # Fallback splitter should respect its chunk_size

        # tests/processing/test_text_processor.py
    def test_chunk_exceeds_hard_threshold_triggers_fallback(self, capsys):
        # hard_split_threshold is the key here.
        # We want recursive_splitter to produce a chunk > hard_split_threshold.
        # Then fallback_splitter should act on that oversized chunk.

        hard_thresh = 20
        # Let recursive_splitter's chunk_size be larger, so it might produce a big chunk.
        # Let fallback_splitter's chunk_size be smaller to show it splitting.
        # The TextProcessor's fallback_splitter uses the same chunk_size as recursive_splitter by default.
        # This means the fallback will only make a difference if its separators (" ", "") are more effective
        # than the recursive_splitter's default separators on the oversized chunk.

        processor = TextProcessor(
            chunk_size=30,  # Target for recursive_splitter AND fallback_splitter
            chunk_overlap=5,
            hard_split_threshold=hard_thresh
        )

        # This text will be one chunk by recursive_splitter (len 34) because no internal primary separators
        # and it's > chunk_size (30), so recursive_splitter will split it into one chunk of 30 and one of 4+overlap.
        # Let's use a simpler text that recursive_splitter will make one chunk of,
        # and that one chunk is > hard_thresh.
        # Example: A sentence with spaces, but short enough that recursive_splitter (chunk_size=30) keeps it as one.
        # But this one chunk is > hard_thresh (20).
        text_to_test = "This single semantic chunk is over twenty characters."  # len 51
        # recursive_splitter (chunk_size=30) will likely split this into:
        # 1. "This single semantic chunk is " (len 30)
        # 2. "nk is over twenty characters." (len 29, with overlap 5 from "chunk")
        # Both of these are > hard_thresh (20).

        chunks = processor.process_and_chunk(text_to_test)
        captured = capsys.readouterr()
        print(f"Captured out: {captured.out}")  # For debugging the messages

        # Check that the INFO message for exceeding threshold was printed for chunks from recursive_splitter
        # that were > hard_thresh
        # Example: "INFO: Chunk (len=30) exceeded hard_split_threshold (20)..."
        # Example: "INFO: Chunk (len=29) exceeded hard_split_threshold (20)..."
        assert f"exceeded hard_split_threshold ({hard_thresh})" in captured.out

        # Now, the chunks in `chunks` are the result of the *fallback_splitter*
        # which also has chunk_size=30 but uses separators [" ", ""].
        # It should have taken the >20 char chunks from recursive_splitter and split them further if possible
        # using spaces, then characters, to try and meet its own chunk_size of 30.
        # Since the fallback_splitter's chunk_size is 30, and the inputs to it were already ~30,
        # it might not split them much further unless they were "word1 word2" and word1 was <30.

        # This test is becoming tricky because both splitters use the same chunk_size.
        # The real value of fallback is if recursive_splitter produces a 500-char paragraph
        # (because chunk_size=600) but hard_split_threshold=100. Then fallback (chunk_size=600,
        # but with " " and "" separators) would break that 500-char para down.

        # Let's simplify the test to focus on the fallback TRIGGER and its effect.
        # We need an input that recursive_splitter makes ONE chunk of, and that chunk > hard_thresh.
        processor_fb_trigger = TextProcessor(
            chunk_size=100,  # Recursive splitter is generous
            chunk_overlap=10,
            hard_split_threshold=50  # Fallback triggers if chunk > 50
        )
        # This will be ONE chunk from recursive_splitter (len=70)
        long_unit_for_fallback = "ThisIsOneVeryLongUnitFromRecursiveSplitterThatShouldTriggerTheFallback"  # len=70

        final_chunks = processor_fb_trigger.process_and_chunk(long_unit_for_fallback)
        captured_final = capsys.readouterr()

        assert f"INFO: Chunk (len={len(long_unit_for_fallback)}) exceeded hard_split_threshold (50)" in captured_final.out

        # The fallback_splitter (chunk_size=100, separators " ", "") was given the 70-char string.
        # Since it has no spaces, it will use "" (char split) to make chunks <= 100.
        # In this case, it will still be one chunk of 70 from the fallback.
        # To see fallback *actually split*, its chunk_size would need to be smaller than the input.
        # This means our TextProcessor's fallback_splitter should ideally have a configurable,
        # potentially smaller chunk_size if its only job is hard breaking.
        # OR, the current fallback_splitter (same chunk_size, but more aggressive separators)
        # will only make a difference if the oversized chunk from recursive_splitter *has* spaces/chars
        # that allow further splitting by the fallback's separators.

        # Let's test that the fallback *does something* if the oversized chunk has spaces
        processor_fb_splits_with_space = TextProcessor(
            chunk_size=100,  # Recursive generous
            chunk_overlap=10,
            hard_split_threshold=50
        )
        # This will be ONE chunk from recursive_splitter (len=60)
        long_unit_with_spaces = "This long unit has spaces and will trigger the fallback now."  # len=60

        final_chunks_ws = processor_fb_splits_with_space.process_and_chunk(long_unit_with_spaces)
        captured_ws = capsys.readouterr()

        assert f"INFO: Chunk (len={len(long_unit_with_spaces)}) exceeded hard_split_threshold (50)" in captured_ws.out
        # fallback_splitter (chunk_size=100, separators " ", "") gets the 60-char string.
        # It will split by space.
        # "This long unit has spaces" (len=27)
        # "and will trigger the" (len=20)
        # "fallback now." (len=13)
        # All these are < 100 (fallback's chunk_size).
        assert len(final_chunks_ws) > 1  # It should have been split by the fallback
        for chunk in final_chunks_ws:
            assert len(chunk) < len(long_unit_with_spaces)  # Each piece is smaller than original
            assert len(chunk) <= processor_fb_splits_with_space.chunk_size  # Fallback respects its own chunk_size

    def test_supercalifragilistic_is_char_split_by_recursive_splitter(self, capsys):
        # Renamed for clarity: RecursiveCharacterTextSplitter with a small chunk_size
        # WILL character-split a long word if "" is in its separators.
        # The hard_split_threshold is not expected to be hit by these small initial chunks.

        processor = TextProcessor(
            chunk_size=10,
            chunk_overlap=2
            # hard_split_threshold is default (e.g., 1000)
            # Default separators include ""
        )
        long_word = "Supercalifragilisticexpialidocious"  # len 34

        chunks = processor.process_and_chunk(long_word)
        captured = capsys.readouterr()

        # Fallback mechanism should NOT be triggered because initial chunks are small
        assert "INFO: Chunk (len=" not in captured.out

        # Assertions based on the actual output from RecursiveCharacterTextSplitter
        # performing character splits due to small chunk_size and "" separator.
        assert len(chunks) == 4
        assert chunks[0] == "Supercalif"
        assert chunks[1] == "ifragilist"
        assert chunks[2] == "sticexpial"
        assert chunks[3] == "alidocious"
