from __future__ import annotations

import argparse
import sys
from typing import Sequence

from pydantic import BaseModel, Field


class ValidationArgs(BaseModel):
    allow_cpu: bool = False
    skip_pgvector: bool = False
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
        description="Validate Python, CUDA, pgvector/psycopg, and WhisperX access."
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow validation to continue when CUDA is unavailable.",
    )
    parser.add_argument(
        "--skip-pgvector",
        action="store_true",
        help="Skip the pgvector/psycopg availability check.",
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
    if not args.whisperx_model.strip():
        raise ValueError("--whisperx-model must not be empty.")


def format_vram_gib(total_memory_bytes: int) -> str:
    return f"{total_memory_bytes / (1024 ** 3):.1f} GiB"


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
        f"{version_label} is not supported. Use Python 3.10 or 3.11 for WhisperX.",
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


def check_pgvector() -> ValidationCheck:
    try:
        import psycopg  # noqa: F401
    except ImportError as error:
        return fail_check(
            "pgvector",
            f"psycopg3 is not installed: {error}.",
            remediation="Install with: pip install 'psycopg[binary]>=3.1' pgvector>=0.2",
        )
    try:
        import pgvector  # noqa: F401
    except ImportError as error:
        return fail_check(
            "pgvector",
            f"pgvector is not installed: {error}.",
            remediation="Install with: pip install pgvector>=0.2",
        )
    return pass_check("pgvector", "psycopg3 and pgvector packages are available.")


def release_gpu_resources(resource: object | None = None) -> None:
    if resource is not None:
        del resource
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    print("[1/4] Checking Python version...")
    summary.checks.append(check_python_version())

    print("[2/4] Checking CUDA availability...")
    summary.checks.append(check_cuda(args.allow_cpu))

    if not args.skip_pgvector:
        print("[3/4] Checking pgvector/psycopg availability...")
        summary.checks.append(check_pgvector())
    else:
        summary.checks.append(pass_check("pgvector", "Skipped pgvector/psycopg check by request."))

    if not args.skip_whisperx:
        print("[4/4] Checking WhisperX model loading...")
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

    print("\nAll environment checks passed. Ready for pipeline execution.")


if __name__ == "__main__":
    main()
