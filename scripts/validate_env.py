from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PYANNOTE_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-3.1"


class ValidationArgs(BaseModel):
    allow_cpu: bool = False
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    skip_pyannote: bool = False
    skip_whisperx: bool = False
    whisperx_model: str = "small"


class ValidationCheck(BaseModel):
    details: str | None = None
    name: str
    remediation: str | None = None
    success: bool


class ValidationSummary(BaseModel):
    checks: list[ValidationCheck] = Field(default_factory=list)

    def has_failures(self) -> bool:
        return any(not check.success for check in self.checks)

    def exit_code(self) -> int:
        return 1 if self.has_failures() else 0


def parse_args(argv: Sequence[str] | None = None) -> ValidationArgs:
    parser = argparse.ArgumentParser(
        description="Validate Python, CUDA, Hugging Face, pyannote, and WhisperX access."
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow validation to continue when CUDA is unavailable.",
    )
    parser.add_argument(
        "--pyannote-model",
        default="pyannote/speaker-diarization-3.1",
        help="pyannote model repo to validate.",
    )
    parser.add_argument(
        "--skip-pyannote",
        action="store_true",
        help="Skip the pyannote gated model access check.",
    )
    parser.add_argument(
        "--skip-whisperx",
        action="store_true",
        help="Skip the WhisperX CUDA model load check.",
    )
    parser.add_argument(
        "--whisperx-model",
        default="small",
        help="WhisperX model size to load for validation.",
    )
    namespace = parser.parse_args(argv)
    return ValidationArgs.model_validate(vars(namespace))


def validate_inputs(args: ValidationArgs) -> None:
    if not args.pyannote_model.strip():
        raise ValueError("--pyannote-model must not be empty.")
    if not args.whisperx_model.strip():
        raise ValueError("--whisperx-model must not be empty.")


def format_vram_gib(total_memory_bytes: int) -> str:
    return f"{total_memory_bytes / (1024 ** 3):.1f} GiB"


def load_hf_token() -> str | None:
    load_dotenv()
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    fallback_token = os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
    return fallback_token or None


def pass_check(name: str, details: str, remediation: str | None = None) -> ValidationCheck:
    return ValidationCheck(name=name, success=True, details=details, remediation=remediation)


def fail_check(name: str, details: str, remediation: str | None = None) -> ValidationCheck:
    return ValidationCheck(name=name, success=False, details=details, remediation=remediation)


def check_python_version() -> ValidationCheck:
    version_info = sys.version_info
    supported = version_info.major == 3 and version_info.minor in {10, 11}
    version_label = f"Python {version_info.major}.{version_info.minor}.{version_info.micro}"
    if supported:
        return pass_check("python", f"{version_label} is supported.")
    return fail_check(
        "python",
        f"{version_label} is not supported. Use Python 3.10 or 3.11 for WhisperX and pyannote.",
        remediation="Create a virtual environment with `py -3.11 -m venv .venv` and rerun validation.",
    )


def check_cuda(allow_cpu: bool) -> ValidationCheck:
    try:
        import torch
    except ImportError as error:
        return fail_check(
            "cuda",
            f"PyTorch is not installed: {error}.",
            remediation="Install the CUDA-enabled PyTorch wheels before running this script.",
        )

    if not torch.cuda.is_available():
        if allow_cpu:
            return pass_check(
                "cuda",
                "CUDA is unavailable. Continuing because --allow-cpu was set.",
                remediation="Install the CUDA-enabled PyTorch wheel to avoid very slow CPU transcription.",
            )
        return fail_check(
            "cuda",
            "CUDA is not available. WhisperX would fall back to CPU and become impractically slow.",
            remediation="Install the CUDA-enabled PyTorch wheel and confirm the NVIDIA driver is current.",
        )

    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return pass_check("cuda", f"Detected {device_name} with {format_vram_gib(total_memory)}.")


def check_hf_token() -> tuple[ValidationCheck, str | None]:
    token = load_hf_token()
    if token:
        return pass_check("hf_token", "Hugging Face token loaded from .env."), token
    return (
        fail_check(
            "hf_token",
            "No Hugging Face token was found in .env.",
            remediation="Copy .env.example to .env and set HF_TOKEN before rerunning validation.",
        ),
        None,
    )


def release_gpu_resources(resource: object | None = None) -> None:
    if resource is not None:
        del resource
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_torch_checkpoint_loading() -> None:
    try:
        import torch
    except ImportError:
        return

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    torch_version_module = getattr(torch, "torch_version", None)
    torch_version_class = getattr(torch_version_module, "TorchVersion", None)
    if callable(add_safe_globals) and torch_version_class is not None:
        add_safe_globals([torch_version_class])

    original_load = getattr(torch, "load", None)
    if not callable(original_load) or getattr(original_load, "_task004_compat", False):
        return

    def compatible_load(*args: object, **kwargs: object) -> object:
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    setattr(compatible_load, "_task004_compat", True)
    setattr(torch, "load", compatible_load)


def check_pyannote_access(model_name: str, token: str | None) -> ValidationCheck:
    if token is None:
        return fail_check(
            "pyannote",
            "Skipped pyannote access because no Hugging Face token is configured.",
            remediation=f"Add HF_TOKEN to .env and accept the model agreement at {PYANNOTE_MODEL_URL}.",
        )

    try:
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
        from pyannote.audio import Pipeline
    except ImportError as error:
        return fail_check(
            "pyannote",
            f"pyannote dependencies are unavailable: {error}.",
            remediation="Install dependencies from requirements.txt before rerunning validation.",
        )

    pipeline: object | None = None
    try:
        configure_torch_checkpoint_loading()
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        return pass_check("pyannote", f"Loaded gated model access for {model_name}.")
    except GatedRepoError:
        return fail_check(
            "pyannote",
            f"Access to {model_name} is gated.",
            remediation=f"Accept the model agreement at {PYANNOTE_MODEL_URL} and retry.",
        )
    except RepositoryNotFoundError as error:
        return fail_check(
            "pyannote",
            f"The pyannote repository was not found: {error}",
            remediation="Confirm the model repo name is correct.",
        )
    except HfHubHTTPError as error:
        status_code = error.response.status_code if error.response is not None else "unknown"
        return fail_check(
            "pyannote",
            f"Hugging Face HTTP error while loading {model_name}: status {status_code}.",
            remediation=f"Verify the token scope and model agreement at {PYANNOTE_MODEL_URL}.",
        )
    except OSError as error:
        return fail_check(
            "pyannote",
            f"pyannote failed to initialize: {error}",
            remediation="Check local cache permissions and internet access, then retry.",
        )
    finally:
        release_gpu_resources(pipeline)


def check_whisperx_model(model_name: str, allow_cpu: bool) -> ValidationCheck:
    try:
        import torch
        import whisperx.asr
    except ImportError as error:
        return fail_check(
            "whisperx",
            f"WhisperX dependencies are unavailable: {error}.",
            remediation="Install dependencies from requirements.txt before rerunning validation.",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        return fail_check(
            "whisperx",
            "WhisperX validation requires CUDA unless --allow-cpu is set.",
            remediation="Fix CUDA first or rerun with --allow-cpu for a degraded validation pass.",
        )

    compute_type = "float16" if device == "cuda" else "int8"
    model: object | None = None
    try:
        model = whisperx.asr.WhisperModel(model_name, device=device, compute_type=compute_type)
        return pass_check("whisperx", f"Loaded WhisperX ASR backend model '{model_name}' on {device}.")
    except RuntimeError as error:
        return fail_check(
            "whisperx",
            f"WhisperX model load failed: {error}",
            remediation="Use the 'small' model on the local RTX 3050 or free GPU memory before retrying.",
        )
    except ValueError as error:
        return fail_check(
            "whisperx",
            f"WhisperX model load failed: {error}",
            remediation="Confirm the WhisperX model name is valid and retry.",
        )
    except OSError as error:
        return fail_check(
            "whisperx",
            f"WhisperX could not download or open model files: {error}",
            remediation="Check internet access and local cache permissions, then retry.",
        )
    finally:
        release_gpu_resources(model)


def run_validation(args: ValidationArgs) -> ValidationSummary:
    summary = ValidationSummary()

    print("[1/5] Checking Python version...")
    python_check = check_python_version()
    summary.checks.append(python_check)

    print("[2/5] Checking CUDA availability...")
    cuda_check = check_cuda(args.allow_cpu)
    summary.checks.append(cuda_check)

    print("[3/5] Checking Hugging Face token...")
    token_check, token = check_hf_token()
    summary.checks.append(token_check)

    if not args.skip_pyannote:
        print("[4/5] Checking pyannote gated model access...")
        summary.checks.append(check_pyannote_access(args.pyannote_model, token))
    else:
        summary.checks.append(pass_check("pyannote", "Skipped pyannote access check by request."))

    if not args.skip_whisperx:
        print("[5/5] Checking WhisperX model loading...")
        summary.checks.append(check_whisperx_model(args.whisperx_model, args.allow_cpu))
    else:
        summary.checks.append(pass_check("whisperx", "Skipped WhisperX model load by request."))

    return summary


def print_summary(summary: ValidationSummary) -> None:
    print("\nValidation summary:")
    for check in summary.checks:
        status = "PASS" if check.success else "FAIL"
        print(f"- [{status}] {check.name}: {check.details}")
        if check.remediation:
            print(f"  remediation: {check.remediation}")


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        validate_inputs(args)
    except ValueError as error:
        print(f"Input validation failed: {error}")
        raise SystemExit(2) from error

    summary = run_validation(args)
    print_summary(summary)
    if summary.has_failures():
        raise SystemExit(summary.exit_code())

    print("\nAll environment checks passed. Ready for TASK-002.")


if __name__ == "__main__":
    main()