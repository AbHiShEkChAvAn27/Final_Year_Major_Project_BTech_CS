"""
Board Auto-Detection Module
Automatically detects connected development boards (Arduino, ESP32, Raspberry Pi)
via USB serial ports and system inspection.
"""

import subprocess
import platform as platform_mod
from pathlib import Path
from typing import List, Dict, Optional


# Known USB Vendor ID / Product ID mappings
KNOWN_BOARDS = {
    # Arduino official
    (0x2341, 0x0043): {"platform": "arduino", "board": "Arduino Uno"},
    (0x2341, 0x0001): {"platform": "arduino", "board": "Arduino Uno"},
    (0x2341, 0x0010): {"platform": "arduino", "board": "Arduino Mega 2560"},
    (0x2341, 0x0042): {"platform": "arduino", "board": "Arduino Mega 2560"},
    (0x2341, 0x8036): {"platform": "arduino", "board": "Arduino Leonardo"},
    (0x2341, 0x8037): {"platform": "arduino", "board": "Arduino Micro"},
    (0x2341, 0x003E): {"platform": "arduino", "board": "Arduino Due"},
    (0x2341, 0x804D): {"platform": "arduino", "board": "Arduino Zero"},
    (0x2341, 0x0058): {"platform": "arduino", "board": "Arduino Nano Every"},
    (0x2341, 0x0070): {"platform": "arduino", "board": "Arduino Nano RP2040"},
    (0x2341, 0x005B): {"platform": "arduino", "board": "Arduino Nano 33 IoT"},
    (0x2341, 0x8057): {"platform": "arduino", "board": "Arduino Nano 33 BLE"},

    # Arduino clones (CH340 / CH341 chipset)
    (0x1A86, 0x7523): {"platform": "arduino", "board": "Arduino-Compatible (CH340)"},
    (0x1A86, 0x5523): {"platform": "arduino", "board": "Arduino-Compatible (CH341)"},

    # FTDI-based Arduino boards
    (0x0403, 0x6001): {"platform": "arduino", "board": "Arduino-Compatible (FTDI)"},
    (0x0403, 0x6015): {"platform": "arduino", "board": "Arduino-Compatible (FTDI)"},

    # ESP32 boards (CP2102 / CP2104)
    (0x10C4, 0xEA60): {"platform": "esp32", "board": "ESP32 (CP2102/CP2104)"},
    (0x10C4, 0xEA70): {"platform": "esp32", "board": "ESP32-S2 (CP2104)"},

    # ESP32-S2/S3 native USB
    (0x303A, 0x1001): {"platform": "esp32", "board": "ESP32-S2"},
    (0x303A, 0x80D1): {"platform": "esp32", "board": "ESP32-S3"},
    (0x303A, 0x1001): {"platform": "esp32", "board": "ESP32-S2"},

    # Raspberry Pi Pico (RP2040) — treated as arduino-compatible
    (0x2E8A, 0x0005): {"platform": "arduino", "board": "Raspberry Pi Pico (RP2040)"},
    (0x2E8A, 0x000A): {"platform": "arduino", "board": "Raspberry Pi Pico W"},
}

# Keywords in device description that indicate board type
DESCRIPTION_HINTS = {
    "arduino": "arduino",
    "esp32": "esp32",
    "esp8266": "esp32",      # treat ESP8266 similar to ESP32
    "cp2102": "esp32",
    "cp2104": "esp32",
    "ch340": "arduino",
    "ch341": "arduino",
    "ftdi": "arduino",
    "mega": "arduino",
    "uno": "arduino",
    "nano": "arduino",
    "leonardo": "arduino",
}


def detect_serial_boards() -> List[Dict]:
    """
    Detect boards connected via USB serial ports using pyserial.
    Returns a list of dicts with keys: platform, board, port, vid, pid
    """
    detected = []
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()

        for port in ports:
            board_info = None

            # 1. Try matching by VID:PID
            if port.vid is not None and port.pid is not None:
                board_info = KNOWN_BOARDS.get((port.vid, port.pid))

            # 2. Fallback: match by description keywords
            if not board_info:
                desc = (port.description or "").lower()
                manufacturer = (port.manufacturer or "").lower()
                combined = f"{desc} {manufacturer}"
                for keyword, plat in DESCRIPTION_HINTS.items():
                    if keyword in combined:
                        board_info = {"platform": plat, "board": f"Detected via description ({keyword})"}
                        break

            if board_info:
                detected.append({
                    "platform": board_info["platform"],
                    "board": board_info["board"],
                    "port": port.device,
                    "vid": hex(port.vid) if port.vid else None,
                    "pid": hex(port.pid) if port.pid else None,
                    "description": port.description,
                })
    except ImportError:
        pass
    except Exception:
        pass

    return detected


def detect_arduino_cli_boards() -> List[Dict]:
    """
    Use arduino-cli (if available) to detect boards.
    Returns a list of dicts with keys: platform, board, port, fqbn
    """
    detected = []
    try:
        result = subprocess.run(
            ["arduino-cli", "board", "list", "--format", "json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            # arduino-cli v0.35+ uses "detected_ports" key
            ports_list = data if isinstance(data, list) else data.get("detected_ports", data.get("serialBoards", []))
            for item in ports_list:
                port_info = item.get("port", item)
                boards = item.get("matching_boards", item.get("boards", []))
                port_address = port_info.get("address", port_info.get("port", ""))
                for board in boards:
                    name = board.get("name", "Unknown Board")
                    fqbn = board.get("fqbn", "")
                    plat = "esp32" if "esp32" in fqbn.lower() else "arduino"
                    detected.append({
                        "platform": plat,
                        "board": name,
                        "port": port_address,
                        "fqbn": fqbn,
                    })
    except FileNotFoundError:
        pass  # arduino-cli not installed
    except Exception:
        pass

    return detected


def detect_raspberry_pi() -> Optional[Dict]:
    """
    Detect if the current machine is a Raspberry Pi.
    Checks device-tree model and /proc/cpuinfo.
    """
    # Method 1: Device tree model file
    model_path = Path("/sys/firmware/devicetree/base/model")
    if model_path.exists():
        try:
            model = model_path.read_text().strip().rstrip('\x00')
            if "raspberry pi" in model.lower():
                return {
                    "platform": "raspberry_pi",
                    "board": model,
                    "port": "local",
                    "description": "Running on this Raspberry Pi",
                }
        except Exception:
            pass

    # Method 2: /proc/cpuinfo
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        if "raspberry pi" in cpuinfo.lower() or "BCM2" in cpuinfo:
            # Extract model from "Model" line
            for line in cpuinfo.splitlines():
                if line.startswith("Model"):
                    model_name = line.split(":", 1)[-1].strip()
                    return {
                        "platform": "raspberry_pi",
                        "board": model_name,
                        "port": "local",
                        "description": "Running on this Raspberry Pi",
                    }
            return {
                "platform": "raspberry_pi",
                "board": "Raspberry Pi (detected via cpuinfo)",
                "port": "local",
                "description": "Running on this Raspberry Pi",
            }
    except Exception:
        pass

    return None


def detect_all_boards() -> List[Dict]:
    """
    Run all detection methods and return a unified, deduplicated list.
    Each entry: { platform, board, port, ... }
    """
    all_boards = []
    seen_ports = set()

    # 1. arduino-cli detection (most accurate names)
    for b in detect_arduino_cli_boards():
        key = b.get("port", "")
        if key and key not in seen_ports:
            seen_ports.add(key)
            all_boards.append(b)

    # 2. Serial port detection
    for b in detect_serial_boards():
        key = b.get("port", "")
        if key and key not in seen_ports:
            seen_ports.add(key)
            all_boards.append(b)

    # 3. Local Raspberry Pi detection
    rpi = detect_raspberry_pi()
    if rpi:
        all_boards.append(rpi)

    return all_boards


def get_platform_from_boards(boards: List[Dict]) -> str:
    """
    Given a list of detected boards, return the best-guess platform string.
    Priority: if multiple boards, pick the first one.
    Returns one of: 'arduino', 'esp32', 'raspberry_pi', or '' if none detected.
    """
    if not boards:
        return ""
    return boards[0]["platform"]


def format_board_summary(boards: List[Dict]) -> str:
    """
    Return a human-readable summary of detected boards.
    """
    if not boards:
        return "No boards detected. Connect a board via USB and click Refresh."

    lines = []
    for i, b in enumerate(boards, 1):
        port = b.get("port", "N/A")
        board_name = b.get("board", "Unknown")
        plat = b.get("platform", "unknown")
        lines.append(f"{i}. **{board_name}** — `{plat}` on `{port}`")
    return "\n".join(lines)
