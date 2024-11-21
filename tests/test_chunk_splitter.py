from lexical_benchmark.stats.block_average2 import chunk_splitter


# Test cases
def test_empty_input():
    """Test the case when the input list is empty."""
    result = chunk_splitter([], chunk_size=1000)
    assert result == []


def test_chunk_size_smaller_than_list():
    """Test when the list has more words than the chunk size."""
    words = ["word" + str(i) for i in range(1, 5001)]
    result = chunk_splitter(words, chunk_size=1000)
    assert len(result) == 5  # 5000 words split into 5 chunks of 1000 words
    assert len(result[0]) == 1000  # Each chunk should have 1000 words
    assert len(result[-1]) == 1000  # Last chunk should have 1000 words


def test_chunk_size_larger_than_list():
    """Test when the chunk size is larger than the number of words in the list."""
    words = ["word" + str(i) for i in range(1, 5001)]
    result = chunk_splitter(words, chunk_size=10000)
    assert result == []  # No full chunks should be returned


def test_exactly_full_chunks():
    """Test when the list can be split into exact full chunks."""
    words = ["word" + str(i) for i in range(1, 32001)]  # 32,000 words
    result = chunk_splitter(words, chunk_size=8000)
    assert len(result) == 4  # 32,000 words split into 4 chunks of 8000 words
    assert len(result[0]) == 8000  # Each chunk should have 8000 words


def test_partial_chunk_discarded():
    """Test when there are leftover words that don't fit into a full chunk."""
    words = ["word" + str(i) for i in range(1, 16001)]  # 16,000 words
    result = chunk_splitter(words, chunk_size=10000)
    assert len(result) == 1  # Only 1 full chunk of size 10,000
    assert len(result[0]) == 10000  # First chunk should have 10,000 words


def test_single_word_chunks():
    """Test when the list has only one word."""
    words = ["word1"]
    result = chunk_splitter(words, chunk_size=1)
    assert len(result) == 1  # One chunk containing the single word
    assert result[0] == ["word1"]  # The chunk should contain the single word


def test_large_chunk_size():
    """Test when the chunk size is very large."""
    words = ["word" + str(i) for i in range(1, 5001)]
    result = chunk_splitter(words, chunk_size=100000)
    assert result == []  # No full chunks should be returned


def test_chunk_size_equals_list_size():
    """Test when the chunk size equals the size of the word list."""
    words = ["word" + str(i) for i in range(1, 5001)]  # 5000 words
    result = chunk_splitter(words, chunk_size=5000)
    assert len(result) == 1  # One chunk containing all words
    assert result[0] == words  # The chunk should be the entire word list
