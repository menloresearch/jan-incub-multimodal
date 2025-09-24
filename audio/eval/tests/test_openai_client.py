import pytest
from openai import OpenAI


@pytest.fixture
def test_audio():
    """Load real test audio file"""
    audio_path = "data/2086-149220-0033.wav"
    with open(audio_path, "rb") as f:
        return f.read()


@pytest.fixture
def expected_text():
    """Expected transcription text"""
    with open("data/2086-149220-0033.txt", "r") as f:
        return f.read().strip()


@pytest.fixture
def openai_client():
    """OpenAI client pointing to local server"""
    return OpenAI(
        api_key="fake-key",  # Server doesn't require real auth
        base_url="http://localhost:8000/v1",
    )


def test_basic_transcription(openai_client, test_audio, expected_text):
    """Test basic transcription with JSON format"""
    response = openai_client.audio.transcriptions.create(
        file=("test.wav", test_audio, "audio/wav"), model="turbo"
    )

    assert hasattr(response, "text")
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    print(f"Expected: {expected_text}")
    print(f"Got: {response.text}")
    # Basic check - should contain some key words
    assert "Phoebe" in response.text or "portrait" in response.text


def test_verbose_json_format(openai_client, test_audio):
    """Test verbose JSON format response"""
    response = openai_client.audio.transcriptions.create(
        file=("test.wav", test_audio, "audio/wav"),
        model="turbo",
        response_format="verbose_json",
    )

    assert hasattr(response, "text")
    assert hasattr(response, "duration")
    assert hasattr(response, "language")
    assert isinstance(response.text, str)
    assert isinstance(response.duration, (int, float))
    assert isinstance(response.language, str)


def test_word_timestamps(openai_client, test_audio):
    """Test word-level timestamps"""
    response = openai_client.audio.transcriptions.create(
        file=("test.wav", test_audio, "audio/wav"),
        model="turbo",
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )

    assert hasattr(response, "text")
    assert hasattr(response, "words")
    assert isinstance(response.words, list)


def test_segment_timestamps(openai_client, test_audio):
    """Test segment-level timestamps"""
    response = openai_client.audio.transcriptions.create(
        file=("test.wav", test_audio, "audio/wav"),
        model="turbo",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )

    assert hasattr(response, "text")
    assert hasattr(response, "segments")
    assert isinstance(response.segments, list)


def test_text_format(openai_client, test_audio):
    """Test plain text format response"""
    response = openai_client.audio.transcriptions.create(
        file=("test.wav", test_audio, "audio/wav"),
        model="turbo",
        response_format="text",
    )

    assert isinstance(response, str)
    assert len(response) > 0
