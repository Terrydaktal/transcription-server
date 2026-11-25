from ..config import RuntimeConfig
from ..consensus.backend import AdaptiveConsensusTranscriber, ConsensusTranscriber
from .canary_qwen import CanaryQwenTranscriber
from .faster_whisper import FasterWhisperTranscriber
from .granite_speech import GraniteSpeechTranscriber
from .parakeet import ParakeetTranscriber


def build_transcriber(config: RuntimeConfig):
    if config.backend == "faster-whisper":
        return FasterWhisperTranscriber(config)
    if config.backend == "canary-qwen":
        return CanaryQwenTranscriber(model_name=config.model_name, device=config.device)
    if config.backend == "parakeet":
        return ParakeetTranscriber(config)
    if config.backend == "granite-speech":
        return GraniteSpeechTranscriber(config)
    if config.backend == "consensus":
        return ConsensusTranscriber(config)
    if config.backend == "adaptive-consensus":
        return AdaptiveConsensusTranscriber(config)
    raise ValueError(f"Unsupported backend: {config.backend}")
