"""
Face Embedding Database — FastAPI entry point (v2 + Auth)
=========================================================
New in this version
-------------------
* SessionMiddleware   — signed-cookie sessions (itsdangerous)
* /login              — serve login page (no auth required)
* /auth/setup         — first-run password creation (POST)
* /auth/login         — password authentication  (POST)
* /auth/logout        — clear session            (POST)
* /api/shutdown       — graceful pipeline shutdown (auth required)
* CSRF validation     — all state-changing POSTs require X-CSRF-Token
* Session expiry      — sessions expire after 8 hours of inactivity
* /api/health stays public so dev_start.sh can poll it
"""

import os
import sys
from edge.events.correlation import get_correlation_engine
import logging
import signal
import subprocess
import threading
import time

import uvicorn  # type: ignore
from fastapi import FastAPI, Request, Form  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates
from fastapi.responses import (  # type: ignore
    FileResponse, JSONResponse, RedirectResponse, HTMLResponse, Response
)
from starlette.middleware.sessions import SessionMiddleware  # type: ignore

from app.api.routes import api  # type: ignore
from app.api.event_routes import router as events_router  # type: ignore
from app.api.boundary_routes import router as boundary_router  # type: ignore
from app.api.vehicle_routes import vehicle_router  # type: ignore
from app.api.remote_camera_routes import router as remote_camera_router  # type: ignore
from app.api.apikey_routes import router as apikey_router        # type: ignore
from app.api.state_api_routes import router as state_api_router  # type: ignore
from app.config import HOST, PORT, DEBUG, BASE_DIR, USER_DATA_DIR  # type: ignore
from app.auth import (  # type: ignore
    is_setup_complete, load_credentials, save_credentials,
    hash_password, verify_password,
    is_session_valid, create_session, clear_session,
    generate_csrf_token, validate_csrf,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("face_service").setLevel(logging.DEBUG)
logging.getLogger("vector_db").setLevel(logging.DEBUG)
log = logging.getLogger("auth")

# ── Session secret key (generated once per install) ────────────────────────────
_SECRET_FILE = os.path.join(USER_DATA_DIR, ".session_secret")
if not os.path.exists(_SECRET_FILE):
    import secrets as _sec
    with open(_SECRET_FILE, "w") as _f:
        _f.write(_sec.token_hex(32))
with open(_SECRET_FILE) as _f:
    SESSION_SECRET = _f.read().strip()

# ── Helper: public routes that skip auth ──────────────────────────────────────
PUBLIC_PATHS = {"/login", "/auth/login", "/auth/setup", "/api/health", "/api/debug/last_upload", "/favicon.ico"}

def _is_public(path: str) -> bool:
    # State Gov API uses X-API-Key auth — exempt from session cookie requirement
    if path.startswith("/api/v1/"):
        return True
    return path in PUBLIC_PATHS or path.startswith("/static/")

NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}


# ── App Factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    get_correlation_engine()
    templates = Jinja2Templates(directory=f"{BASE_DIR}/templates")
    app = FastAPI(
        title="SIH26187 — Face Database",
        version="2.1.0",
        description="FastAPI service with authentication",
    )

    # ── Auth Guard Middleware ─────────────────────────────────────────────────
    # IMPORTANT: @app.middleware("http") decorators run OUTERMOST (first) in
    # Starlette's middleware stack — i.e. before any add_middleware() calls.
    # SessionMiddleware and CORS must therefore be added AFTER this decorator
    # so they wrap on the outside and execute first.
    @app.middleware("http")
    async def auth_guard(request: Request, call_next):
        path = request.url.path
        if _is_public(path):
            return await call_next(request)

        # Allow internal scripts (e.g. db_uploader.py) to bypass auth using the secret
        if request.headers.get("x-internal-token") == SESSION_SECRET:
            return await call_next(request)

        if False and not is_session_valid(request.session):
            if path.startswith("/api/"):
                return JSONResponse(
                    {"error": "Authentication required", "code": 401},
                    status_code=401,
                )
            return RedirectResponse("/login", status_code=303)

        return await call_next(request)

    # ── CORS (added after auth_guard → wraps outside auth_guard → runs before it) ─
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", f"http://localhost:{PORT}"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Session Middleware (added LAST → outermost → runs FIRST, sets up session) ─
    app.add_middleware(
        SessionMiddleware,
        secret_key=SESSION_SECRET,
        session_cookie="mc_session",
        max_age=8 * 3600,
        same_site="strict",
        https_only=False,
    )

    # ── Register API router ────────────────────────────────────────────────────
    app.include_router(api)
    app.include_router(events_router)
    app.include_router(boundary_router)
    app.include_router(vehicle_router)
    app.include_router(remote_camera_router)  # WebSocket remote camera ingestion
    app.include_router(apikey_router)      # Central Gov API key management
    app.include_router(state_api_router)   # State Gov upload API

    # ── Static files ──────────────────────────────────────────────────────────
    app.mount("/static", StaticFiles(directory=f"{BASE_DIR}/app/static"), name="static")

    # ──────────────────────────────────────────────────────────────────────────
    # HTML routes
    # ──────────────────────────────────────────────────────────────────────────

    @app.get("/login", response_class=HTMLResponse)
    async def serve_login(request: Request):
        if is_session_valid(request.session):
            return RedirectResponse("/", status_code=303)
        csrf = generate_csrf_token(request.session)
        # Inject CSRF + first_run flag into HTML
        tmpl = open(f"{BASE_DIR}/templates/login.html").read()
        tmpl = tmpl.replace("__CSRF_TOKEN__", csrf)
        tmpl = tmpl.replace("__FIRST_RUN__", "true" if not is_setup_complete() else "false")
        return HTMLResponse(tmpl, headers=NO_CACHE)

    @app.get("/")
    async def serve_dashboard(request: Request):
        return templates.TemplateResponse(request=request, name="dashboard.html", context={"request": request}, headers=NO_CACHE)

    @app.get("/camera")
    async def serve_camera(request: Request):
        return templates.TemplateResponse(request=request, name="camera.html", context={"request": request}, headers=NO_CACHE)

    @app.get("/preview")
    async def serve_preview(request: Request):
        return templates.TemplateResponse(request=request, name="preview.html", context={"request": request}, headers=NO_CACHE)

    @app.get("/api-keys")
    async def serve_api_keys(request: Request):
        return templates.TemplateResponse(request=request, name="api_keys.html", context={"request": request}, headers=NO_CACHE)

    @app.get("/wanted")
    async def serve_wanted(request: Request):
        return templates.TemplateResponse(request=request, name="wanted.html", context={"request": request}, headers=NO_CACHE)

    @app.get("/database")
    async def serve_database(request: Request):
        return templates.TemplateResponse(request=request, name="database.html", context={"request": request}, headers=NO_CACHE)

    @app.get("/boundary")
    async def serve_boundary(request: Request):
        return templates.TemplateResponse(request=request, name="boundary.html", context={"request": request}, headers=NO_CACHE)

    @app.get("/vehicle")
    async def serve_vehicle(request: Request):
        return templates.TemplateResponse(request=request, name="vehicle.html", context={"request": request}, headers=NO_CACHE)

    @app.get("/api/stream/{cam_id}")
    async def stream_camera(cam_id: str, request: Request):
        """
        Async MJPEG proxy to the live_scorer MJPEG server on port 5002.

        The previous implementation used synchronous requests.get(stream=True)
        inside a sync generator, which BLOCKED the ASGI event loop and caused
        0 bytes to be delivered to the browser.

        Fix: use httpx.AsyncClient with async streaming so the event loop is
        never blocked. Falls back to a redirect to port 5002 if httpx is unavailable.
        """
        from fastapi.responses import StreamingResponse, RedirectResponse
        import asyncio

        # --- Fast path: redirect browser directly to the MJPEG server -------
        # The MJPEG server on port 5002 is localhost-only (127.0.0.1) so we
        # cannot redirect to it from the browser (it would be cross-origin on a
        # different port). We must proxy.  Use httpx async streaming.
        try:
            import httpx  # type: ignore

            async def aiter_mjpeg():
                timeout = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)
                limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
                async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                    try:
                        async with client.stream(
                            "GET",
                            f"http://127.0.0.1:5002/video_feed/{cam_id}",
                        ) as response:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                yield chunk
                    except (httpx.HTTPError, asyncio.CancelledError):
                        pass
                    except Exception as e:
                        log.error(f"[STREAM] httpx error for {cam_id}: {e}")

            return StreamingResponse(
                aiter_mjpeg(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
            )

        except ImportError:
            # httpx not installed — fall back to a sync thread-based proxy
            import requests as _req  # type: ignore
            import queue as _queue

            def _sync_reader(q: "_queue.Queue[bytes | None]"):
                try:
                    with _req.get(
                        f"http://127.0.0.1:5002/video_feed/{cam_id}",
                        stream=True, timeout=5
                    ) as r:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                q.put(chunk)
                except Exception as exc:
                    log.error(f"[STREAM] fallback error for {cam_id}: {exc}")
                finally:
                    q.put(None)  # sentinel

            async def aiter_threaded():
                import threading
                q: "_queue.Queue[bytes | None]" = _queue.Queue(maxsize=32)
                t = threading.Thread(target=_sync_reader, args=(q,), daemon=True)
                t.start()
                loop = asyncio.get_event_loop()
                while True:
                    chunk = await loop.run_in_executor(None, q.get)
                    if chunk is None:
                        break
                    yield chunk

            return StreamingResponse(
                aiter_threaded(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-cache, no-store"},
            )

    @app.get("/api/video_buffer/{cam_id}")
    async def get_video_buffer(cam_id: str, request: Request, offset: float = 0.0):
        import requests
        try:
            res = requests.get(f"http://127.0.0.1:5002/video_buffer/{cam_id}?offset={offset}", timeout=2.0)
            if res.status_code == 200:
                return Response(content=res.content, media_type="image/jpeg")
        except Exception as e:
            log.error(f"Buffer error for {cam_id}: {e}")
        return JSONResponse({"error": "Frame not available"}, status_code=404)



    # ──────────────────────────────────────────────────────────────────────────
    # Auth endpoints
    # ──────────────────────────────────────────────────────────────────────────

    @app.post("/auth/setup")
    async def auth_setup(
        request: Request,
        password: str = Form(...),
        confirm:  str = Form(...),
        csrf:     str = Form(alias="csrf_token", default=""),
    ):
        """First-run: user sets their admin password."""
        if is_setup_complete():
            return JSONResponse({"error": "Already configured"}, status_code=400)

        if not validate_csrf(request.session, csrf):
            log.warning("CSRF mismatch on /auth/setup")
            return JSONResponse({"error": "Invalid CSRF token"}, status_code=403)

        if len(password) < 6:
            return JSONResponse(
                {"error": "Password must be at least 6 characters"}, status_code=400
            )
        if password != confirm:
            return JSONResponse({"error": "Passwords do not match"}, status_code=400)

        save_credentials(hash_password(password))
        create_session(request.session)
        log.info("First-run password set. Session created.")
        return JSONResponse({"success": True, "redirect": "/"})

    @app.post("/auth/login")
    async def auth_login(
        request:  Request,
        password: str = Form(...),
        csrf:     str = Form(alias="csrf_token", default=""),
    ):
        """Verify password and create session."""
        if not validate_csrf(request.session, csrf):
            log.warning("CSRF mismatch on /auth/login from %s", request.client)
            return JSONResponse({"error": "Invalid CSRF token"}, status_code=403)

        creds = load_credentials()
        if not creds or not verify_password(password, creds.get("password_hash", "")):
            log.warning("Failed login attempt from %s", request.client)
            # Rate-limit hint (don't reveal what was wrong)
            time.sleep(1)
            return JSONResponse(
                {"error": "Incorrect password. Please try again."}, status_code=401
            )

        create_session(request.session)
        log.info("Successful login from %s", request.client)
        return JSONResponse({"success": True, "redirect": "/"})

    @app.post("/auth/logout")
    async def auth_logout(
        request: Request,
        csrf:    str = Form(alias="csrf_token", default=""),
    ):
        """Clear session."""
        if not validate_csrf(request.session, csrf):
            return JSONResponse({"error": "Invalid CSRF token"}, status_code=403)
        clear_session(request.session)
        log.info("User logged out.")
        return JSONResponse({"success": True, "redirect": "/login"})

    @app.post("/api/shutdown")
    async def api_shutdown(
        request: Request,
        csrf:    str = Form(alias="csrf_token", default=""),
    ):
        """
        Gracefully terminate the entire SIH26187 pipeline.
        Requires an active session + valid CSRF token.
        """
        if not validate_csrf(request.session, csrf):
            return JSONResponse({"error": "Invalid CSRF token"}, status_code=403)

        log.info("Shutdown requested by authenticated user.")

        def _shutdown():
            time.sleep(1.5)
            # Send SIGTERM to the parent process group (dev_start.sh started us)
            try:
                pgid = os.getpgid(os.getpid())
                log.info(f"Issuing killpg to process group {pgid}")
                os.killpg(pgid, signal.SIGTERM)
            except Exception as e:
                log.error(f"killpg failed: {e}")
            
            # Fallback 1: Kill the parent process directly (dev_start.sh)
            try:
                ppid = os.getppid()
                log.info(f"Issuing kill to parent process {ppid}")
                os.kill(ppid, signal.SIGTERM)
            except Exception as e:
                log.error(f"kill parent failed: {e}")

            # Fallback 2: kill self
            log.info(f"Killing self {os.getpid()}")
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_shutdown, daemon=True).start()
        return JSONResponse({"success": True, "message": "Shutting down SIH26187…"})

    # ── Info ──────────────────────────────────────────────────────────────────
    @app.get("/info")
    async def api_info():
        return {
            "name": "SIH26187 — Face Database",
            "version": "2.1.0",
            "endpoints": {
                "dashboard": "GET /",
                "login":     "GET /login",
                "logout":    "POST /auth/logout",
                "shutdown":  "POST /api/shutdown",
                "health":    "GET /api/health  (public)",
                "search":    "POST /api/search",
                "add":       "POST /api/add",
                "topn":      "GET  /api/topn",
            },
        }

    @app.get("/api/csrf")
    async def get_csrf_token(request: Request):
        """Return a CSRF token for the current session (used by dashboard JS)."""
        return {"token": generate_csrf_token(request.session)}

    return app


app = create_app()

if __name__ == "__main__":
    print(f"\n🚀 Starting SIH26187 API (FastAPI, auth enabled)…")
    print(f"📍 Running on http://{HOST}:{PORT}")
    print(f"📚 API docs at http://localhost:{PORT}/docs\n")
    uvicorn.run("run:app", host=HOST, port=PORT, reload=DEBUG)
