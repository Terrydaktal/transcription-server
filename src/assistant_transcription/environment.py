import ctypes
import os
import sysconfig
from pathlib import Path


def configure_shared_model_caches() -> None:
    """Use the user's global model caches unless explicitly overridden."""
    cache_home = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    os.environ.setdefault("HF_HOME", str(cache_home / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache_home / "torch"))


def preload_local_cuda_libraries() -> None:
    nvidia_root = Path(sysconfig.get_path("purelib")) / "nvidia"
    nvidia_libs = [
        str(path)
        for path in (nvidia_root / "cublas" / "lib", nvidia_root / "cudnn" / "lib")
        if path.is_dir()
    ]
    if not nvidia_libs:
        return
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = ":".join(nvidia_libs)
    os.environ["LD_LIBRARY_PATH"] = f"{new_ld}:{existing_ld}" if existing_ld else new_ld

    for library_path in (
        nvidia_root / "cublas" / "lib" / "libcublas.so.12",
        nvidia_root / "cudnn" / "lib" / "libcudnn.so.9",
    ):
        if library_path.exists():
            try:
                ctypes.CDLL(str(library_path))
            except OSError:
                pass


configure_shared_model_caches()
preload_local_cuda_libraries()
