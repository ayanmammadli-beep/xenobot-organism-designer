"""
backend/api/main.py
━━━━━━━━━━━━━━━━━━━
FastAPI application — all REST + WebSocket endpoints.

Run with:  uvicorn backend.api.main:app --reload --port 8000
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
import json
import uuid
import os
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

app = FastAPI(
    title="Morpheus API",
    description="EDA suite for programmable organism design",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (use Redis in production)
SESSIONS: Dict[str, Dict] = {}


# ═══════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════

class ProblemInput(BaseModel):
    problem_description: str = Field(
        ..., example="Design an organism to clear PFAS from groundwater at pH 6.8, 15°C, flow 2cm/s"
    )
    explicit_params: Optional[Dict[str, float]] = Field(
        default=None,
        example={"viscosity_mPas": 1.1, "temperature_C": 15, "pH": 6.8, "flow_speed_cms": 2.0}
    )
    optimization_config: Optional[Dict[str, Any]] = Field(
        default={
            "n_generations": 30,
            "bo_iterations": 15,
            "grad_steps": 80,
            "pop_size": 20,
            "grid_size": [8, 8, 6]
        }
    )
    api_key: Optional[str] = Field(default=None, description="Anthropic API key")


class WetLabUpload(BaseModel):
    session_id: str
    simulated_capture_pct: float
    actual_capture_pct: float
    actual_pH: Optional[float]
    actual_temp_C: Optional[float]
    microscopy_notes: Optional[str]
    additional_data: Optional[Dict] = {}


class SimulateRequest(BaseModel):
    grid: List[List[List[int]]]
    fluid_key: str = "water_pure"
    contaminant_key: str = "hdpe_microplastic"
    flow_speed_cms: float = 1.0
    n_contaminants: int = 40


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "service": "morpheus-api", "version": "1.0.0"}


@app.get("/api/fluids")
async def list_fluids():
    """Return all available fluid models."""
    from data.contaminants.database import FLUID_DATABASE
    return {k: {"name": v.name, "model_type": v.model_type,
                "density": v.density, "pH": v.pH}
            for k, v in FLUID_DATABASE.items()}


@app.get("/api/contaminants")
async def list_contaminants():
    """Return all available contaminant types."""
    from data.contaminants.database import CONTAMINANT_DATABASE
    return {k: {"name": v.name, "category": v.category,
                "diameter_um": v.diameter*1e6, "interaction": v.interaction_model,
                "color": v.color_hex}
            for k, v in CONTAMINANT_DATABASE.items()}


@app.post("/api/session/create")
async def create_session(problem: ProblemInput):
    """Create a new design session. Returns session_id."""
    session_id = str(uuid.uuid4())[:8]
    SESSIONS[session_id] = {
        "id":          session_id,
        "problem":     problem.problem_description,
        "params":      problem.explicit_params or {},
        "opt_config":  problem.optimization_config,
        "api_key":     problem.api_key,
        "status":      "created",
        "results":     {},
    }
    return {"session_id": session_id, "status": "created"}


@app.post("/api/simulate")
async def simulate_grid(req: SimulateRequest):
    """
    Run a single simulation for a given voxel grid.
    Used for quick interactive testing before full optimization.
    """
    from data.contaminants.database import FLUID_DATABASE, CONTAMINANT_DATABASE
    from backend.physics.simulator import SimulationConfig, run_simulation
    import numpy as np

    grid = np.array(req.grid, dtype=int)
    fluid       = FLUID_DATABASE.get(req.fluid_key, FLUID_DATABASE["water_pure"])
    contaminant = CONTAMINANT_DATABASE.get(req.contaminant_key, CONTAMINANT_DATABASE["hdpe_microplastic"])

    config = SimulationConfig(
        fluid             = fluid,
        contaminant       = contaminant,
        flow_velocity     = np.array([req.flow_speed_cms * 0.01, 0.0, 0.0]),
        n_contaminants    = req.n_contaminants,
        sim_steps         = 300,
        record_history    = True,
    )

    result = run_simulation(grid, config, record_history=True)

    return {
        "fitness":             result.fitness,
        "capture_efficiency":  result.capture_efficiency,
        "energy_spent_nJ":     result.energy_spent * 1e9,
        "dimensionless":       result.dimensionless,
        "capture_rate_curve":  result.capture_rate_curve,
        "history_length":      len(result.history),
        "history":             result.history[:5],  # first 5 frames
    }


@app.post("/api/wetlab/upload")
async def upload_wetlab(data: WetLabUpload):
    """Upload wet lab results and trigger discrepancy analysis."""
    if data.session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")

    session = SESSIONS[data.session_id]
    from backend.agent.scientist import ToolExecutor

    executor = ToolExecutor(session)
    result = executor.execute("analyze_wetlab_discrepancy", {
        "simulated_capture_pct": data.simulated_capture_pct,
        "actual_capture_pct":    data.actual_capture_pct,
        "wetlab_conditions": {
            "actual_pH":    data.actual_pH,
            "actual_temp_C": data.actual_temp_C,
        },
        "microscopy_notes": data.microscopy_notes or "",
        "additional_data":  data.additional_data or {},
    })

    session["wetlab_analysis"] = result
    return result


@app.get("/api/session/{session_id}/report")
async def get_report(session_id: str):
    """Get the complete design report for a session."""
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    session = SESSIONS[session_id]
    return {
        "session_id": session_id,
        "problem":    session["problem"],
        "status":     session["status"],
        "results":    session.get("results", {}),
    }


@app.get("/api/session/{session_id}/history")
async def get_history(session_id: str):
    """Get evolutionary history (generation log) for a session."""
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    return SESSIONS[session_id].get("generation_log", [])


# ═══════════════════════════════════════════════════════════
# WEBSOCKET — Real-time progress streaming
# ═══════════════════════════════════════════════════════════

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time optimization progress.

    Message protocol:
      Client → {"action": "start_optimization"} | {"action": "run_agent", "message": "..."}
      Server → {"type": "progress"|"agent_text"|"tool_call"|"result"|"error"|"done", ...}
    """
    await websocket.accept()

    if session_id not in SESSIONS:
        await websocket.send_json({"type": "error", "content": "Session not found"})
        await websocket.close()
        return

    session = SESSIONS[session_id]

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            # ── Run agent ──────────────────────────────────
            if action == "run_agent":
                msg     = data.get("message", session["problem"])
                api_key = data.get("api_key") or session.get("api_key") or os.getenv("ANTHROPIC_API_KEY")

                if not api_key:
                    await websocket.send_json({
                        "type": "error",
                        "content": "No API key. Set ANTHROPIC_API_KEY env var or pass in request."
                    })
                    continue

                from backend.agent.scientist import run_agent

                for event in run_agent(msg, api_key, session):
                    await websocket.send_json(event)
                    await asyncio.sleep(0)  # yield to event loop

            # ── Run optimization (without agent) ───────────
            elif action == "start_optimization":
                await _run_optimization_ws(websocket, session, data)

            # ── Run single simulation ───────────────────────
            elif action == "simulate":
                grid = np.array(data["grid"], dtype=int)
                await _run_sim_ws(websocket, session, grid)

    except WebSocketDisconnect:
        pass


async def _run_optimization_ws(ws: WebSocket, session: Dict, config: Dict):
    """Run optimization and stream progress over WebSocket."""
    from data.contaminants.database import FLUID_DATABASE, CONTAMINANT_DATABASE
    from backend.physics.simulator import SimulationConfig, run_simulation
    from backend.generative.optimizer import (
        MorphologyVAE, MorpheusSurrogate, BayesianMorphologyOptimizer
    )

    fluid_key   = session.get("fluid_key", "water_pure")
    contam_key  = session.get("contaminant_key", "hdpe_microplastic")
    fluid       = FLUID_DATABASE.get(fluid_key, FLUID_DATABASE["water_pure"])
    contaminant = CONTAMINANT_DATABASE.get(contam_key, CONTAMINANT_DATABASE["hdpe_microplastic"])
    grid_size   = tuple(config.get("grid_size", [8, 8, 6]))

    sim_config = SimulationConfig(
        fluid=fluid, contaminant=contaminant,
        flow_velocity=np.array([session.get("flow_speed", 0.01), 0.0, 0.0]),
        grid_size=grid_size, sim_steps=300, n_contaminants=40,
    )

    vae       = MorphologyVAE(grid_size=grid_size)
    surrogate = MorpheusSurrogate()

    def sim_fn(grid, record_history=False, generation=0):
        return run_simulation(grid, sim_config, record_history=record_history, generation=generation)

    optimizer = BayesianMorphologyOptimizer(vae, surrogate, sim_fn, sim_config, grid_size)

    async def progress_cb(info):
        await ws.send_json({"type": "progress", "data": {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                                          for k, v in info.items()
                                                          if k != "grid"}})

    # Run in thread pool to avoid blocking event loop
    import asyncio
    loop = asyncio.get_event_loop()

    gen_log_buffer = []
    def sync_cb(info):
        gen_log_buffer.append(info)

    results = await loop.run_in_executor(
        None,
        lambda: optimizer.optimize(
            evo_gens   = config.get("n_generations", 30),
            bo_iters   = config.get("bo_iterations", 15),
            grad_steps = config.get("grad_steps", 80),
            pop_size   = config.get("pop_size", 20),
            progress_callback = sync_cb,
        )
    )

    # Stream buffered progress
    for info in gen_log_buffer:
        await ws.send_json({"type": "progress", "data": {
            k: (v.tolist() if hasattr(v, 'tolist') else v)
            for k, v in info.items() if k != "grid"
        }})

    session["results"]        = {"best_fitness": results["best_fitness"]}
    session["generation_log"] = optimizer.generation_log
    session["best_grid"]      = results["best_grid"].tolist() if results["best_grid"] is not None else None
    session["status"]         = "completed"

    final_sim = results.get("final_simulation")
    await ws.send_json({
        "type": "optimization_complete",
        "data": {
            "best_fitness":        results["best_fitness"],
            "capture_efficiency":  final_sim.capture_efficiency if final_sim else None,
            "energy_nJ":           final_sim.energy_spent * 1e9 if final_sim else None,
            "best_grid":           results["best_grid"].tolist() if results["best_grid"] is not None else None,
            "n_generations":       len(optimizer.generation_log),
            "stages": [
                {"stage": s["stage"], "fitness": s["fitness"]}
                for s in results.get("stages", [])
            ],
        }
    })


async def _run_sim_ws(ws: WebSocket, session: Dict, grid: np.ndarray):
    from data.contaminants.database import FLUID_DATABASE, CONTAMINANT_DATABASE
    from backend.physics.simulator import SimulationConfig, run_simulation
    import asyncio

    fluid       = FLUID_DATABASE["water_pure"]
    contaminant = CONTAMINANT_DATABASE["hdpe_microplastic"]
    config      = SimulationConfig(fluid=fluid, contaminant=contaminant, sim_steps=300)
    loop        = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        None, lambda: run_simulation(grid, config, record_history=True))

    for frame in result.history:
        await ws.send_json({"type": "sim_frame", "data": frame})
        await asyncio.sleep(0.05)

    await ws.send_json({
        "type": "sim_complete",
        "data": {
            "fitness":            result.fitness,
            "capture_efficiency": result.capture_efficiency,
            "energy_nJ":          result.energy_spent * 1e9,
        }
    })
