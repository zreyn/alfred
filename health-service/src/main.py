"""
Health Status API for Homepage Dashboard

A FastAPI service providing system health metrics including CPU, RAM, disk, GPU,
Docker container, and LiveKit status.
"""

import os
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any

import docker
import httpx
import jwt
import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure psutil to use host paths when running in Docker
PROC_PATH = os.environ.get("PROC_PATH", "/proc")
HOST_ROOT = os.environ.get("HOST_ROOT", "/host/root")

# LiveKit configuration
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "http://localhost:7880")

# Docker compose project name for filtering containers
COMPOSE_PROJECT = os.environ.get("COMPOSE_PROJECT", "alfred")

# Set psutil's PROCFS_PATH for host metrics
if PROC_PATH != "/proc":
    psutil.PROCFS_PATH = PROC_PATH


def get_cpu_status() -> dict[str, Any]:
    """Get CPU usage and information."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_freq = psutil.cpu_freq()
    cpu_count = psutil.cpu_count()
    cpu_count_logical = psutil.cpu_count(logical=True)

    return {
        "usage_percent": cpu_percent,
        "frequency_mhz": cpu_freq.current if cpu_freq else None,
        "cores_physical": cpu_count,
        "cores_logical": cpu_count_logical,
    }


def get_ram_status() -> dict[str, Any]:
    """Get RAM usage information."""
    mem = psutil.virtual_memory()

    return {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "usage_percent": mem.percent,
    }


def get_disk_status() -> dict[str, Any]:
    """Get disk usage information for root partition."""
    # Use host root path if running in Docker with mounted filesystem
    disk_path = HOST_ROOT if os.path.exists(HOST_ROOT) else "/"
    disk = psutil.disk_usage(disk_path)

    return {
        "total_gb": round(disk.total / (1024**3), 2),
        "used_gb": round(disk.used / (1024**3), 2),
        "free_gb": round(disk.free / (1024**3), 2),
        "usage_percent": round(disk.percent, 1),
    }


def get_gpu_status() -> dict[str, Any]:
    """Get GPU status using nvidia-smi if available."""
    if not shutil.which("nvidia-smi"):
        return {"available": False, "error": "nvidia-smi not found"}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return {"available": False, "error": result.stderr.strip()}

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                memory_total = int(parts[1])
                memory_used = int(parts[2])
                memory_util_percent = round(memory_used / memory_total * 100, 1) if memory_total > 0 else 0
                compute_util = int(parts[4])
                memory_bandwidth_util = int(parts[5])

                # Use memory bandwidth utilization as a better indicator of GPU activity
                # Falls back to compute utilization if memory bandwidth is 0
                utilization = memory_bandwidth_util if memory_bandwidth_util > 0 else compute_util

                gpus.append(
                    {
                        "name": parts[0],
                        "memory_total_mb": memory_total,
                        "memory_used_mb": memory_used,
                        "memory_free_mb": int(parts[3]),
                        "memory_utilization_percent": memory_util_percent,
                        "utilization_percent": utilization,
                        "compute_utilization_percent": compute_util,
                        "memory_bandwidth_utilization_percent": memory_bandwidth_util,
                        "temperature_c": int(parts[6]),
                    }
                )

        return {"available": True, "gpus": gpus}

    except subprocess.TimeoutExpired:
        return {"available": False, "error": "nvidia-smi timed out"}
    except Exception as e:
        return {"available": False, "error": str(e)}


def get_container_status() -> dict[str, Any]:
    """Get Docker container status for Alfred project containers."""
    try:
        client = docker.from_env()
        # Filter containers by compose project label
        containers = client.containers.list(
            all=True,
            filters={"label": f"com.docker.compose.project={COMPOSE_PROJECT}"},
        )

        container_list = []
        running_count = 0
        unhealthy_count = 0

        for container in containers:
            status = container.status
            health = None

            # Get health status if available
            if container.attrs.get("State", {}).get("Health"):
                health = container.attrs["State"]["Health"].get("Status")

            if status == "running":
                running_count += 1
            if health == "unhealthy":
                unhealthy_count += 1

            container_list.append(
                {
                    "name": container.name,
                    "status": status,
                    "health": health,
                }
            )

        # Sort by name for consistent ordering
        container_list.sort(key=lambda x: x["name"])

        return {
            "total": len(containers),
            "running": running_count,
            "unhealthy": unhealthy_count,
            "containers": container_list,
        }

    except Exception as e:
        return {"error": str(e), "total": 0, "running": 0, "unhealthy": 0, "containers": []}


def get_livekit_status() -> dict[str, Any]:
    """Get LiveKit server status via API."""
    try:
        # Generate a JWT token for LiveKit API access
        now = int(time.time())
        claims = {
            "iss": LIVEKIT_API_KEY,
            "sub": LIVEKIT_API_KEY,
            "nbf": now,
            "exp": now + 300,  # 5 minute expiry
            "video": {"roomList": True},
        }
        token = jwt.encode(claims, LIVEKIT_API_SECRET, algorithm="HS256")

        # Call LiveKit's ListRooms API
        with httpx.Client(timeout=5.0) as client:
            response = client.post(
                f"{LIVEKIT_URL}/twirp/livekit.RoomService/ListRooms",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={},
            )

        if response.status_code != 200:
            return {"available": False, "error": f"HTTP {response.status_code}"}

        data = response.json()
        rooms = data.get("rooms", [])

        total_participants = sum(room.get("numParticipants", 0) for room in rooms)

        return {
            "available": True,
            "rooms": len(rooms),
            "participants": total_participants,
        }

    except httpx.ConnectError:
        return {"available": False, "error": "Cannot connect to LiveKit"}
    except Exception as e:
        return {"available": False, "error": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: prime CPU percent measurement
    psutil.cpu_percent(interval=None)
    yield


app = FastAPI(
    title="Alfred Health API",
    description="System health status for Homepage dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "healthy"}


@app.get("/status")
async def system_status() -> dict[str, Any]:
    """
    Get comprehensive system health status.

    Returns CPU, RAM, disk, and GPU metrics for Homepage dashboard.
    """
    return {
        "cpu": get_cpu_status(),
        "ram": get_ram_status(),
        "disk": get_disk_status(),
        "gpu": get_gpu_status(),
    }


@app.get("/status/cpu")
async def cpu_status() -> dict[str, Any]:
    """Get CPU status only."""
    return get_cpu_status()


@app.get("/status/ram")
async def ram_status() -> dict[str, Any]:
    """Get RAM status only."""
    return get_ram_status()


@app.get("/status/disk")
async def disk_status() -> dict[str, Any]:
    """Get disk status only."""
    return get_disk_status()


@app.get("/status/gpu")
async def gpu_status() -> dict[str, Any]:
    """Get GPU status only."""
    return get_gpu_status()


@app.get("/status/containers")
async def containers_status() -> dict[str, Any]:
    """Get Docker container status for Alfred project."""
    return get_container_status()


@app.get("/status/livekit")
async def livekit_status() -> dict[str, Any]:
    """Get LiveKit server status."""
    return get_livekit_status()
