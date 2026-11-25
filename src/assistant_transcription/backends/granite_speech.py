import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from ..config import DEFAULT_GRANITE_PROMPT, RuntimeConfig
from ..log_output import service_log as print


@contextmanager
def prepare_granite_audio(audio_path: str):
    """Normalize Granite input to channel 0, mono 16 kHz PCM16 WAV."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
        output_path = output_file.name
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        audio_path,
        "-af",
        "pan=mono|c0=c0",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Granite audio preprocessing failed: {result.stderr.strip()}")
        print(
            "Granite audio prepared: "
            "sample_rate=16000 channels=1 selected_channel=0 encoding=pcm_s16le "
            f"preprocess_ms={round((time.perf_counter() - started) * 1000)}"
        )
        yield output_path
    finally:
        Path(output_path).unlink(missing_ok=True)


class GraniteSpeechTranscriber:
    def __init__(self, config: RuntimeConfig) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "Granite Speech Q8 requires HTTPX. Install it with "
                "`uv sync --extra granite-speech`."
            ) from exc

        self.config = config
        self.model_name = config.model_name
        self.prompt = self._build_prompt()
        self.base_url = (
            f"http://{config.granite_server_host}:{config.granite_server_port}"
        )
        self.client = httpx.Client(
            timeout=httpx.Timeout(config.granite_request_timeout_seconds, connect=10.0)
        )
        self.process: subprocess.Popen | None = None
        self._start_server()
        print(f"Granite Speech Q8 prompt: {self.prompt}")
        print("Granite Speech Q8 llama-server is ready.")

    def _build_prompt(self) -> str:
        prompt = self.config.granite_prompt.strip() or DEFAULT_GRANITE_PROMPT
        key_phrases_file = self.config.parakeet_key_phrases_file.strip()
        self.keywords = []
        if key_phrases_file:
            phrases_path = Path(key_phrases_file).expanduser()
            if not phrases_path.is_file():
                raise FileNotFoundError(f"Granite Speech keyword file not found: {phrases_path}")
            phrases = [
                line.strip()
                for line in phrases_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if phrases:
                self.keywords = phrases
                prompt = f"{prompt.rstrip()} Keywords: {', '.join(phrases)}"
        return prompt

    def _server_command(self, binary: str) -> list[str]:
        command = [
            binary,
            "-hf",
            self.model_name,
            "--host",
            self.config.granite_server_host,
            "--port",
            str(self.config.granite_server_port),
            "--parallel",
            "1",
            "--ctx-size",
            str(self.config.granite_context_size),
        ]
        if self.config.device.startswith("cuda"):
            command.extend(("--n-gpu-layers", "auto"))
        return command

    def _is_ready(self) -> bool:
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def _start_server(self) -> None:
        if self._is_ready():
            print(f"Using existing Granite Speech llama-server at {self.base_url}.")
            return

        binary = shutil.which(self.config.granite_server_binary)
        if binary is None:
            raise RuntimeError(
                f"Granite Speech Q8 requires '{self.config.granite_server_binary}'. "
                "Install the CachyOS package with `sudo pacman -S llama-cpp`."
            )
        command = self._server_command(binary)
        print(
            f"Starting Granite Speech Q8 llama-server: model='{self.model_name}' "
            f"url='{self.base_url}' context_size={self.config.granite_context_size}"
        )
        self.process = subprocess.Popen(command)
        deadline = time.monotonic() + self.config.granite_server_startup_timeout_seconds
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                self.process = None
                raise RuntimeError(
                    f"Granite Speech llama-server exited during startup with code {return_code}."
                )
            if self._is_ready():
                return
            time.sleep(0.25)
        self.close()
        raise TimeoutError(
            "Granite Speech llama-server did not become ready within "
            f"{self.config.granite_server_startup_timeout_seconds:g} seconds."
        )

    def transcribe_file(self, audio_path: str) -> str:
        temperature = f"{self.config.granite_temperature:g}"
        max_tokens = str(self.config.granite_max_new_tokens)
        print(
            "Granite transcription request: "
            f"prompt={self.prompt!r} max_tokens={max_tokens!r} temperature={temperature!r}"
        )
        with prepare_granite_audio(audio_path) as prepared_path:
            with open(prepared_path, "rb") as audio_file:
                response = self.client.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    data={
                        "model": self.model_name,
                        "prompt": self.prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    files={"file": ("granite-input.wav", audio_file, "audio/wav")},
                )
        if response.is_error:
            raise RuntimeError(
                f"Granite Speech llama-server returned HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        payload = response.json()
        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"Granite Speech response did not contain text: {payload}")
        return text.strip()

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None:
            self.process = None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        client = getattr(self, "client", None)
        if client is not None:
            client.close()
