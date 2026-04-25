"""
Client for connecting to the SOCRATES environment.

IMPORTANT: This file NEVER imports from server/ — clean client/server separation.
Uses only models.py (shared types) and websockets for communication.
"""

import asyncio
import json
import logging
from typing import Optional

import websockets

from models import SocratesAction, SocratesObservation, SocratesState

logger = logging.getLogger(__name__)


class SocratesEnv:
    """
    Client for the SOCRATES environment.
    
    Connects to the server via WebSocket. Provides both sync and async APIs.
    Never imports from server/ — uses only shared models.
    """

    def __init__(self, base_url: str = "ws://localhost:7860/ws"):
        """
        Initialize the client.

        Args:
            base_url: WebSocket URL of the environment server.
        """
        self.url = base_url
        self._ws = None
        self._loop = None

    # ─── Async API ────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        if self._ws is None or self._ws.close_code is not None:
            try:
                self._ws = await websockets.connect(self.url)
                logger.info(f"Connected to SOCRATES environment at {self.url}")
            except Exception as e:
                raise ConnectionError(
                    f"Failed to connect to SOCRATES environment at {self.url}: {e}"
                )

    async def _send_and_receive(self, message: dict) -> dict:
        """Send a message and wait for response."""
        await self._connect()
        try:
            await self._ws.send(json.dumps(message))
            response = await self._ws.recv()
            return json.loads(response)
        except websockets.exceptions.ConnectionClosed as e:
            self._ws = None
            raise ConnectionError(f"WebSocket connection lost: {e}")
        except Exception as e:
            raise ConnectionError(f"Communication error: {e}")

    async def async_reset(self, task: str = "foundation") -> SocratesObservation:
        """
        Async reset — start a new tutoring episode.

        Args:
            task: Difficulty level — "foundation", "intermediate", "advanced"
                  (or aliases: "easy", "medium", "hard")

        Returns:
            Initial SocratesObservation.
        """
        message = {"type": "reset", "data": {"task": task}}
        response = await self._send_and_receive(message)

        if "error" in response or response.get("type") == "error":
            error_msg = response.get("error") or response.get("data", {}).get("message", "")
            raise ValueError(f"Reset failed: {error_msg}")

        response_data = response.get("data", {})
        obs_data = response_data.get("observation", response_data)
        return SocratesObservation(**obs_data)

    async def async_step(
        self, action: SocratesAction
    ) -> tuple[SocratesObservation, float, bool, dict]:
        """
        Async step — submit a Socratic question.

        Args:
            action: SocratesAction with the question.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        message = {"type": "step", "data": action.model_dump()}
        response = await self._send_and_receive(message)

        if "error" in response or response.get("type") == "error":
            error_msg = response.get("error") or response.get("data", {}).get("message", "")
            raise RuntimeError(f"Step failed: {error_msg}")

        response_data = response.get("data", {})
        obs_data = response_data.get("observation", response_data)
        observation = SocratesObservation(**obs_data)
        reward = float(response_data.get("reward", 0.0))
        done = bool(response_data.get("done", False))
        info = response_data.get("info", {})

        return observation, reward, done, info

    async def async_state(self) -> SocratesState:
        """Async state — get current environment state for debugging."""
        message = {"type": "state", "data": {}}
        response = await self._send_and_receive(message)

        if "error" in response or response.get("type") == "error":
            error_msg = response.get("error") or response.get("data", {}).get("message", "")
            raise RuntimeError(f"State failed: {error_msg}")

        state_data = response.get("data", {})
        return SocratesState(**state_data)

    async def async_close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None and self._ws.close_code is None:
            await self._ws.close()
            self._ws = None
            logger.info("Disconnected from SOCRATES environment")

    # ─── Sync API ─────────────────────────────────────────────────────────

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def reset(self, task: str = "foundation") -> SocratesObservation:
        """Sync reset — start a new tutoring episode."""
        return self._get_loop().run_until_complete(self.async_reset(task))

    def step(
        self, action: SocratesAction
    ) -> tuple[SocratesObservation, float, bool, dict]:
        """Sync step — submit a Socratic question."""
        return self._get_loop().run_until_complete(self.async_step(action))

    def state(self) -> SocratesState:
        """Sync state — get current environment state."""
        return self._get_loop().run_until_complete(self.async_state())

    def close(self) -> None:
        """Close the WebSocket connection."""
        self._get_loop().run_until_complete(self.async_close())

    # ─── Context Managers ─────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.async_close()
