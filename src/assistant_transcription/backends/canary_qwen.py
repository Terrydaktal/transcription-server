from ..log_output import service_log as print


class CanaryQwenTranscriber:
    def __init__(self, model_name: str, device: str) -> None:
        try:
            import torch
            from nemo.collections.speechlm2.models import SALM
        except ImportError as exc:
            raise RuntimeError(
                "Canary-Qwen requires NeMo ASR. Install it with "
                "`uv sync --extra canary` or `uv pip install 'nemo-toolkit[asr]'`."
            ) from exc

        self.model_name = model_name
        self.device = device
        self.torch = torch
        print(
            f"Loading Canary-Qwen model '{self.model_name}' via NeMo ASR on requested device "
            f"'{self.device}'..."
        )
        print(
            f"torch.cuda.is_available()={torch.cuda.is_available()} "
            f"torch.cuda.device_count()={torch.cuda.device_count()}"
        )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Canary-Qwen was requested on CUDA but torch.cuda.is_available() is false."
            )
        self.model = SALM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        first_param = next(self.model.parameters(), None)
        param_device = str(first_param.device) if first_param is not None else "unknown"
        print(f"Canary-Qwen first parameter device: {param_device}")
        print("Canary-Qwen model loaded successfully.")

    def transcribe_file(self, audio_path: str) -> str:
        answer_ids = self.model.generate(
            prompts=[
                [
                    {
                        "role": "user",
                        "content": f"Transcribe the following: {self.model.audio_locator_tag}",
                        "audio": [audio_path],
                    }
                ]
            ],
            max_new_tokens=256,
        )
        if answer_ids is None:
            return ""

        if hasattr(answer_ids, "numel"):
            if answer_ids.numel() == 0:
                return ""
            first_answer = answer_ids[0] if answer_ids.ndim > 1 else answer_ids
        else:
            if len(answer_ids) == 0:
                return ""
            first_answer = answer_ids[0]

        if hasattr(first_answer, "cpu"):
            first_answer = first_answer.cpu()
        if hasattr(first_answer, "tolist"):
            first_answer = first_answer.tolist()

        return self.model.tokenizer.ids_to_text(first_answer).strip()
