"""
common/io_utils.py
------------------
General-purpose I/O and logging utilities used across
Livinit pipeline modules (scene, layout, usdz).
"""

import os
import json
import shutil
import tempfile
import requests
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union, Optional


# -------------------------------------------------------
# Logging utilities
# -------------------------------------------------------

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """Configure logging with optional file output"""
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            *([] if not log_file else [logging.FileHandler(log_file)])
        ]
    )


def log(message: str, level: str = "INFO") -> None:
    """Unified logging with emoji support"""
    logger = logging.getLogger("livinit")
    level_map = {
        "INFO": logging.INFO,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "DEBUG": logging.DEBUG,
        "OK": logging.INFO
    }
    logger.log(level_map.get(level, logging.INFO), message)


# -------------------------------------------------------
# JSON / file utilities
# -------------------------------------------------------

def read_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Read JSON file with UTF-8 encoding"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    pretty: bool = True
) -> None:
    """Write JSON file with UTF-8 encoding and optional pretty printing"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(
            data, 
            f,
            indent=2 if pretty else None,
            ensure_ascii=False
        )


def copy_dir(src: str, dst: str, overwrite: bool = True):
    if os.path.exists(dst) and overwrite:
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    log(f"Copied directory: {src} → {dst}", "INFO")


def make_temp_dir(prefix: str = "livinit_") -> str:
    tmp = tempfile.mkdtemp(prefix=prefix)
    log(f"Created temp directory: {tmp}", "INFO")
    return tmp


def cleanup_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        log(f"Removed temp directory: {path}", "WARN")


# -------------------------------------------------------
# Networking / downloads
# -------------------------------------------------------

def download_file(url: str, dest_dir: str) -> str:
    """Downloads file if not present in dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.basename(url.split("?")[0])
    out = os.path.join(dest_dir, fname)
    if os.path.exists(out):
        return out

    log(f"Downloading: {url}", "INFO")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    log(f"Downloaded → {out}", "OK")
    return out


# -------------------------------------------------------
# Misc helpers
# -------------------------------------------------------

def safe_name(name: str) -> str:
    """Normalize unsafe filenames or prim names."""
    n = name.strip().replace("-", "_").replace(".", "_").replace(" ", "_")
    if n and n[0].isdigit():
        n = "asset_" + n
    return n


def format_bytes(num: int) -> str:
    """Readable file sizes."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024.0:
            return f"{num:.2f}{unit}"
        num /= 1024.0
    return f"{num:.2f}TB"
