"""Base agent class for the OceanMech-Agent multi-agent system.

All agents in the system inherit from :class:`BaseAgent` and follow a
common perception–reasoning–action loop.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class AgentStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    WAITING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class AgentMessage:
    """Message passed between agents."""

    sender: str
    recipient: str
    subject: str
    payload: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all OceanMech agents.

    Subclasses must implement:
    - :meth:`perceive` – gather observations from the environment / other agents
    - :meth:`reason`   – analyse observations and decide on an action
    - :meth:`act`      – execute the decided action

    The :meth:`run` method orchestrates one full perception–reasoning–action
    cycle and returns a result dict that is forwarded to the orchestrator.
    """

    def __init__(self, name: str, verbose: bool = False) -> None:
        self.name = name
        self.status = AgentStatus.IDLE
        self._inbox: list[AgentMessage] = []
        self._outbox: list[AgentMessage] = []
        self._logger = logging.getLogger(f"oceanmech.agent.{name}")
        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def receive(self, message: AgentMessage) -> None:
        """Deliver an incoming message to this agent's inbox."""
        self._inbox.append(message)

    def send(self, recipient: str, subject: str, payload: dict[str, Any] | None = None) -> AgentMessage:
        """Create and queue an outgoing message."""
        msg = AgentMessage(
            sender=self.name,
            recipient=recipient,
            subject=subject,
            payload=payload or {},
        )
        self._outbox.append(msg)
        return msg

    def flush_outbox(self) -> list[AgentMessage]:
        """Return and clear all queued outgoing messages."""
        messages = list(self._outbox)
        self._outbox.clear()
        return messages

    def flush_inbox(self) -> list[AgentMessage]:
        """Return and clear all queued incoming messages."""
        messages = list(self._inbox)
        self._inbox.clear()
        return messages

    # ------------------------------------------------------------------
    # Core loop (template method pattern)
    # ------------------------------------------------------------------

    @abstractmethod
    def perceive(self, context: dict[str, Any]) -> dict[str, Any]:
        """Extract relevant observations from the shared context.

        Args:
            context: Shared context dict (ocean state, model params, etc.).

        Returns:
            Observations relevant to this agent's task.
        """

    @abstractmethod
    def reason(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Analyse observations and decide on an action plan.

        Args:
            observations: Output of :meth:`perceive`.

        Returns:
            Decision/action plan as a dictionary.
        """

    @abstractmethod
    def act(self, decision: dict[str, Any]) -> dict[str, Any]:
        """Execute the decided action and return results.

        Args:
            decision: Output of :meth:`reason`.

        Returns:
            Result dict that is forwarded to the orchestrator.
        """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute one full perception–reasoning–action cycle.

        Args:
            context: Shared context dict.

        Returns:
            Result dict from :meth:`act`.
        """
        self.status = AgentStatus.RUNNING
        self._logger.debug("%s: perceiving", self.name)
        observations = self.perceive(context)
        self._logger.debug("%s: reasoning", self.name)
        decision = self.reason(observations)
        self._logger.debug("%s: acting", self.name)
        result = self.act(decision)
        self.status = AgentStatus.COMPLETED
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, status={self.status.name})"
