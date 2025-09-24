### What this server is supposed to do
- take in an audio file and return a transcript
- transcribe using faster-whisper

### Features
- vllm whisper for local inference (API will call vLLM)
- In the future API will also route to remote services according to the user request


### out of scope
- eval (handled in the eval sub folder)
- text normalization (will be part of eval folder) (for now) -> the goal is to return raw transcription.
