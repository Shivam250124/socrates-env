"""
FastAPI application for the SOCRATES environment.

Exposes the environment via HTTP and WebSocket endpoints.
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from models import SocratesAction, SocratesObservation, SocratesState
from server.environment import SocratesEnvironment

logger = logging.getLogger(__name__)

# Initialize environment at module level
env = SocratesEnvironment()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SOCRATES Environment server starting...")
    yield
    logger.info("SOCRATES Environment server shutting down.")


app = FastAPI(
    title="SOCRATES: Socratic Teaching Agent RL Environment",
    description=(
        "Train LLMs to teach like Socrates — through questions, never answers. "
        "An RL environment where the agent asks questions to guide a simulated student "
        "from misconception to understanding."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── HTTP Endpoints ──────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def root():
    """Landing page with environment info."""
    return """
    <html>
    <head><title>SOCRATES Environment</title></head>
    <body style="font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 0 20px;">
        <h1>🏛️ SOCRATES: Socratic Teaching Agent</h1>
        <p><strong>Train LLMs to teach like Socrates — through questions, never answers.</strong></p>
        
        <h2>Endpoints</h2>
        <ul>
            <li><code>POST /reset</code> — Start a new episode</li>
            <li><code>POST /step</code> — Submit a Socratic question</li>
            <li><code>GET /state</code> — Get current environment state</li>
            <li><code>GET /health</code> — Health check</li>
            <li><code>GET /tasks</code> — Available tasks</li>
            <li><code>WS /ws</code> — WebSocket interface</li>
            <li><a href="/docs">/docs</a> — Interactive API docs</li>
        </ul>

        <h2>Concept Bank</h2>
        <p>8 programming misconceptions: floating point, recursion, mutable defaults, 
        zero indexing, boolean operators, integer division, pass by reference, negative modulo.</p>

        <h2>Reward Signals (5)</h2>
        <ol>
            <li><strong>Teaching Progress (40%)</strong> — Did the student actually learn?</li>
            <li><strong>Socratic Compliance (25%)</strong> — Did you reveal the answer?</li>
            <li><strong>Question Quality (15%)</strong> — Open-ended vs yes/no</li>
            <li><strong>Efficiency (10%)</strong> — Fewer steps bonus (gated by compliance)</li>
            <li><strong>Misconception Targeting (10%)</strong> — Hitting the right weak spot</li>
        </ol>
    </body>
    </html>
    """


@app.post("/reset")
def reset(task: str = "foundation"):
    """Reset the environment and start a new episode."""
    try:
        obs = env.reset(task=task)
        return {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": False,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: SocratesAction):
    """Execute one step in the environment."""
    try:
        obs = env.step(action)
        reward = getattr(obs, "_reward", 0.0)
        done = getattr(obs, "_done", False)
        info = getattr(obs, "_info", {})

        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def get_state():
    """Return current environment state (for debugging)."""
    return env.state().model_dump()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": "socrates-teaching-env",
        "version": "1.0.0",
        "concepts_loaded": len(env.concept_bank.concepts),
    }


@app.get("/tasks")
def tasks():
    """Return available tasks/difficulty levels."""
    return {
        "tasks": [
            {
                "id": "foundation",
                "name": "Foundation Socratic Teaching",
                "difficulty": "easy",
                "concepts": ["index_zero", "integer_division"],
                "max_steps": 8,
            },
            {
                "id": "intermediate",
                "name": "Intermediate Socratic Teaching",
                "difficulty": "medium",
                "concepts": ["boolean_operators", "modulo_negative", "mutable_defaults"],
                "max_steps": 10,
            },
            {
                "id": "advanced",
                "name": "Advanced Socratic Teaching",
                "difficulty": "hard",
                "concepts": ["floating_point", "recursive_termination", "pass_by_reference"],
                "max_steps": 12,
            },
        ]
    }


@app.get("/grader")
def grader():
    """Return grading/reward information."""
    return {
        "reward_signals": {
            "teaching_progress": {"weight": 0.40, "description": "Understanding delta"},
            "socratic_compliance": {"weight": 0.25, "description": "Answer reveal penalty"},
            "question_quality": {"weight": 0.15, "description": "Open-ended vs yes/no"},
            "efficiency": {"weight": 0.10, "description": "Steps bonus (gated by compliance)"},
            "misconception_targeting": {"weight": 0.10, "description": "Targeting accuracy"},
        },
        "hard_penalties": {
            "min_length": "< 10 chars → -0.5",
            "no_question_mark": "missing ? → -0.4",
            "too_long": "> 200 chars → -0.2",
            "repeated": "cosine > 0.85 → -0.3",
        },
    }


# ─── WebSocket Endpoint ──────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket interface for persistent sessions."""
    await ws.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await ws.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "")
            msg_data = message.get("data", {})

            if msg_type == "reset":
                task = msg_data.get("task", "foundation")
                obs = env.reset(task=task)
                response = {
                    "type": "observation",
                    "data": {
                        "observation": obs.model_dump(),
                        "reward": 0.0,
                        "done": False,
                    },
                }

            elif msg_type == "step":
                action = SocratesAction(**msg_data)
                obs = env.step(action)
                reward = getattr(obs, "_reward", 0.0)
                done = getattr(obs, "_done", False)
                info = getattr(obs, "_info", {})

                response = {
                    "type": "observation",
                    "data": {
                        "observation": obs.model_dump(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                    },
                }

            elif msg_type == "state":
                state = env.state()
                response = {
                    "type": "state",
                    "data": state.model_dump(),
                }

            else:
                response = {
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"},
                }

            await ws.send_text(json.dumps(response, default=str))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "data": {"message": str(e)},
            }))
        except Exception:
            pass
