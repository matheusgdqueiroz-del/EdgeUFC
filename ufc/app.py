"""Desktop app: a local web UI served by the standard library and shown in a chromeless Edge/Chrome window.

Double-click "UFC Edge" (shortcut) -> pythonw -m ufc.app
The server stops by itself a couple of minutes after the window is closed.
"""
import json, os, re, shutil, subprocess, sys, threading, time, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
UI = Path(__file__).resolve().parent / "app_ui"
PRED = ROOT / "predictions" / "latest.json"
SETTINGS = ROOT / "predictions" / "settings.json"
REPORT = ROOT / "reports" / "ufc_edge_audit.html"
PORT = 8765
DEFAULT_SETTINGS = {"bankroll": 1000, "currency": "$", "staking": "steady", "days": 30, "odds_format": "decimal"}

state = {"last_ping": time.time(), "job": None}
lock = threading.Lock()


def load_settings():
    try:
        return {**DEFAULT_SETTINGS, **json.loads(SETTINGS.read_text(encoding="utf8"))}
    except (OSError, ValueError):
        return dict(DEFAULT_SETTINGS)


def public_settings(s):
    return dict(s)


def python_exe():
    exe = Path(sys.executable)
    console = exe.with_name("python.exe")  # pythonw cannot pipe output reliably; the child runs windowless anyway
    return str(console if console.exists() else exe)


def start_job(mode):
    with lock:
        if state["job"] and state["job"]["running"]:
            return state["job"]
        s = load_settings()
        cmd = [python_exe(), "-u", "-m", "ufc.predict", "--days", str(int(s["days"])), "--staking", "steady"]
        if mode == "full":
            cmd.append("--update")
        job = {"mode": mode, "running": True, "started": time.time(), "finished": None, "lines": [], "error": None}
        state["job"] = job
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)}

    def worker():
        try:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            tail = []
            for line in p.stdout:
                line = line.rstrip()
                if not line:
                    continue
                tail.append(line)
                if line.startswith(("[step]", "[update]")):
                    job["lines"].append(line)
            code = p.wait()
            if code != 0:
                job["error"] = "\n".join(tail[-12:])
        except Exception as ex:  # surfaced in the UI
            job["error"] = str(ex)
        finally:
            job["running"] = False
            job["finished"] = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return job


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode("utf8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0]
        state["last_ping"] = time.time()
        if path in ("/", "/index.html"):
            return self._send(200, (UI / "index.html").read_bytes(), "text/html")
        if path == "/icon.png":
            return self._send(200, (UI / "icon.png").read_bytes(), "image/png")
        if path == "/report":
            if REPORT.exists():
                return self._send(200, REPORT.read_bytes(), "text/html")
            return self._send(404, "Report not built yet", "text/plain")
        if path == "/api/state":
            pred = None
            if PRED.exists():
                try:
                    pred = json.loads(PRED.read_text(encoding="utf8"))
                except ValueError:
                    pred = None
            job = state["job"]
            return self._send(200, {"predictions": pred, "settings": public_settings(load_settings()), "job": job, "report": REPORT.exists()})
        if path == "/api/ping":
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        state["last_ping"] = time.time()
        if path == "/api/run":
            mode = self._body().get("mode", "full")
            return self._send(200, start_job("full" if mode == "full" else "quick"))
        if path == "/api/settings":
            s = load_settings()
            b = self._body()
            if "bankroll" in b:
                try:
                    s["bankroll"] = max(0.0, float(b["bankroll"]))
                except (TypeError, ValueError):
                    return self._send(400, {"error": "Bankroll must be a number, for example 1000."})
            if b.get("currency") in ("$", "€", "£", "R$"):
                s["currency"] = b["currency"]
            if b.get("staking") in ("steady", "growth"):
                s["staking"] = b["staking"]
            if b.get("days") in (7, 14, 30, 45):
                s["days"] = b["days"]
            if b.get("odds_format") in ("decimal", "american"):
                s["odds_format"] = b["odds_format"]
            SETTINGS.parent.mkdir(exist_ok=True)
            SETTINGS.write_text(json.dumps(s, indent=1), encoding="utf8")
            return self._send(200, public_settings(s))
        return self._send(404, {"error": "not found"})


def open_window(url):
    candidates = [shutil.which("msedge"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Microsoft\Edge\Application\msedge.exe", shutil.which("chrome"),
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    for c in candidates:
        if c and Path(c).exists():
            subprocess.Popen([c, f"--app={url}", "--window-size=1400,920"], creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
            return
    webbrowser.open(url)


def watchdog(server):
    while True:
        time.sleep(15)
        job = state["job"]
        idle = time.time() - state["last_ping"]
        if idle > 120 and not (job and job["running"]):
            server.shutdown()
            return


def main():
    url = f"http://127.0.0.1:{PORT}/"
    try:
        urlopen(url + "api/ping", timeout=1)  # already running: just show another window
        open_window(url)
        return
    except OSError:
        pass
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=watchdog, args=(server,), daemon=True).start()
    open_window(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
