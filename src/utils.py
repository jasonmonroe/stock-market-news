# src/utils.py

import os
import platform
import time

import psutil
import tensorflow as tf
import torch

from src import config


def start_timer() -> float:
    """Start a timer"""
    return time.time()


def get_time(start_time_float: float) -> str:
    """Formats the elapsed time into a human-readable string."""
    diff = abs(time.time() - start_time_float)
    _, remainder = divmod(diff, config.SECS_IN_MIN * config.SECS_IN_MIN)
    minutes, seconds = divmod(remainder, config.SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * config.MSEC
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"


def show_timer(start_time_float: float) -> None:
    """Prints the elapsed time."""
    print(f"\nRun Time: {get_time(start_time_float)}")


def show_banner(title: str, section: str = '') -> None:
    """Prints a stylized banner for console readability."""
    padding = 2
    strlen = len(title) + padding
    line = '+-' + '-' * strlen + '-+'

    print('')
    print(line)
    print('|  ' + title.upper() + '  |')
    print(line)

    if section:
        print('| ' + section)

    print('')


def show_performance(perf=None, title: str = '') -> None:
    """Displays a performance DataFrame with a banner."""
    if not perf.empty:
        show_banner(title, 'Performance')
        print(f'{perf}')


def show_hardware_info():
    """Displays information about the available CPU and GPU hardware."""
    show_banner('🖥 CPU Info')
    print(f'You have {os.cpu_count()} CPU cores available.')
    print(f"Processor: {platform.processor()}")
    print(f"System RAM: {psutil.virtual_memory().total / (config.KBYTE ** 3):.2f} GB")

    print('')
    show_banner('🎮 GPU Info')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        device_props = torch.cuda.get_device_properties(device_id)

        print(f"Device ID: {device_id}")
        print(f"GPU Name: {device_name}")
        print(f"Total VRAM: {device_props.total_memory / (config.KBYTE ** 3):.2f} GB")
    elif torch.backends.mps.is_available():
        print("✅ Apple Metal (MPS) GPU detected.")
    else:
        print("⚠️ No GPU detected.")
