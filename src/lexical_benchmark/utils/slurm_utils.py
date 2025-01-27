import argparse
import contextlib
import os
import platform
import sys
import time
import typing as t
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread

import humanize
import psutil
import tap
try:
    import torch
except ImportError:
    torch = None


class ProgressTask:
    """Progress Status file for current Slurm job."""

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time."""
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        """Calculate current rate."""
        elapsed = self.elapsed_time
        return self.current / elapsed if elapsed > 0 else 0

    @property
    def eta(self) -> str | None:
        """Compute current ETA."""
        if self.total is None:
            return None
        return f"{((self.total - self.current) / self.rate):.0f}s"

    @property
    def percentage(self) -> str:
        """Compute percentage of completion of task."""
        return f"{self.current / self.total:.1%}" if self.total is not None else "**%"

    def __init__(
        self,
        total: int | None = None,
        update_interval: int = 10,
        task_name: str = "task",
        *,
        to_file: bool = True,
        appends: bool = True,
        target_file: Path | None = None,
    ) -> None:
        """Initialize progress tracker."""
        self.total = total
        self.task_name = task_name
        self.current = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        self.update_interval = update_interval
        self.to_file = to_file
        self.appends = appends

        # Get SLURM job ID or use 'local' if not in SLURM
        self.job_id = os.environ.get("SLURM_JOB_ID", "local")
        if target_file is None:
            self.log_file = Path.cwd() / f"{task_name}_{self.job_id}.progress"
        else:
            self.log_file = target_file.with_name(f"{target_file.stem}_{self.job_id}.progress")
        self.log_file.touch(exist_ok=True)

    def _write_msg(self, progress_msg: str) -> None:
        """Write message to corresponding output."""
        if self.to_file and self.appends:
            self.log_file.safe_append_text(progress_msg + "\n")
        elif self.to_file and not self.appends:
            self.log_file.safe_write_text(progress_msg)
        else:
            print(progress_msg, flush=True)

    def update(self, n: int = 1, status: str | None = None) -> None:
        """Update progress counter and potentially write to output."""
        self.current += n
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            total = self.total if self.total else "∞"
            progress_msg = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{self.task_name}: {self.current}/{total} ({self.percentage}) "
                f"Elapsed: {humanize.precisedelta(timedelta(seconds=self.elapsed_time))} "
            )
            if self.total is None:
                progress_msg += f"Rate: {self.rate:.1f} items/s"

            if status:
                progress_msg += f" | {status}"

            self._write_msg(progress_msg)
            self.last_update = current_time

    def _parallel_update(self, status: str | None = None) -> None:
        """Update without a count."""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            progress_msg = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{self.task_name}: "
                f"Elapsed: {humanize.precisedelta(timedelta(seconds=self.elapsed_time))} "
            )
            if status:
                progress_msg += f" | {status}"
            self._write_msg(progress_msg)
            self.last_update = current_time

    def complete(self, status: str | None = None) -> None:
        """Complete a progress."""
        total = self.total
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed
        if self.total is None:
            total = self.current

        progress_msg = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{self.task_name} Completed {self.current}/{total} (100%) "
            f"Elapsed: {humanize.precisedelta(timedelta(seconds=self.elapsed_time))} "
            f"Rate: {rate:.1f} items/s"
        )
        if status:
            progress_msg += f" | {status}"

        # Write message
        self._write_msg(progress_msg)

    def _parallel_complete(self, status: str | None = None) -> None:
        """Update without a count."""
        current_time = time.time()
        progress_msg = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{self.task_name} Completed "
            f"Elapsed: {humanize.precisedelta(timedelta(seconds=self.elapsed_time))} "
        )
        if status:
            progress_msg += f" | {status}"
        self._write_msg(progress_msg)
        self.last_update = current_time

    def iter_progress(self, it: t.Iterable[t.Any]) -> t.Iterable[t.Any]:
        """Progress of an iterable."""
        self.total = None
        for item in it:
            yield item
            self.update()

    def sequence_progress(self, seq: t.Sequence[t.Any]) -> t.Iterable[t.Any]:
        """Progress on a sequence of items."""
        self.total = len(seq)
        for item in seq:
            yield item
            self.update()

    @contextlib.contextmanager
    def parallel_progress(self, status: str | None = None) -> t.Iterator[None]:
        """Progress running in parallel thread without counter."""
        stop_threads = False

        def status_updater() -> None:
            """Update status."""
            while True:
                time.sleep(self.update_interval)
                self._parallel_update(status)
                if stop_threads:
                    break
            self._parallel_complete(status)

        worker = Thread(target=status_updater)
        worker.daemon = True
        worker.start()

        yield None

        stop_threads = True
        worker.join()


def info_header() -> None:
    """Print Generic information about current slurm job."""
    if os.environ.get("SLURM_JOB_ID") is None:
        return

    torch_info = "Torch was not installed !!"
    if torch:
        torch_info = f"""GPU Info: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}"""

    print(
        f"""
-----------------------------------------------------------
RUNNING JOB {os.environ.get('SLURM_JOB_ID', '-')}
RUNNING Via SLURM on {platform.node()}

SLURM Details:
Job Name: {os.environ.get('SLURM_JOB_NAME', '-')}
Array Task ID: {os.environ.get('SLURM_ARRAY_TASK_ID', '-')}
Partition: {os.environ.get('SLURM_JOB_PARTITION', '-')}
Num Tasks: {os.environ.get('SLURM_NTASKS', '-')}
Num CPUs: {os.environ.get('SLURM_CPUS_ON_NODE', '-')}
GPU Count: {os.environ.get('SLURM_GPUS_ON_NODE', '-')}
Working Directory: {Path.cwd()}

System Details:
CPU Count: {os.cpu_count()}
Memory Info: {psutil.virtual_memory().total / (1024**3):.2f} GB
{torch_info}

Python Environment:
Python: {sys.version} - {sys.executable}
Start Time: {datetime.now()}
-----------------------------------------------------------
""",
        flush=True,
    )


def info_footer() -> None:
    """Print Job Completed Succesfully Status."""
    if os.environ.get("SLURM_JOB_ID") is None:
        return

    print(
        f"""
-----------------------------------------------------------
JOB {os.environ.get('SLURM_JOB_ID', '-')} Completed Running
End Time: {datetime.now()}
-----------------------------------------------------------
""",
        flush=True,
    )


def info_args(args: argparse.Namespace | tap.Tap, separator: str = "-", width: int = 30) -> None:
    """Print command line arguments."""
    data = args.as_dict() if isinstance(args, tap.Tap) else dict(vars(args))

    print(separator * width, flush=True)
    print("#### Arguments passed", flush=True)

    for key, value in data.items():
        print(f"{key}: {value}", flush=True)

    print(separator * width, flush=True)
