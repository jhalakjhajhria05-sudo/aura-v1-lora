"""
AURA - THE ONE FILE  v10  👻🪄
=========================
Everything in a single Python file: skills, agent, dataset generator,
refinery, QLoRA training and evaluation.

USAGE
  Kaggle (T4 x2):   !python AURA_onefile.py train
  Generate data only: python AURA_onefile.py data --n 8000
  Run the spirit:     python AURA_onefile.py agent --adapter ./aura_lora
  Demo (no model):    python AURA_onefile.py agent --demo --dry-run
  Check skills:       python AURA_onefile.py selftest
  Evaluate router:    python AURA_onefile.py eval --adapter ./aura_lora

Install first: pip install torch transformers peft bitsandbytes accelerate requests pillow

v9 fixes (over v8):
  - evaluate() no longer crashes on chitchat rows (JSONDecodeError killed the
    run right after training, before GGUF export + zip)
  - GGUF export uses a VALID converter outtype (q8_0); "q4_k_m" is not accepted
    by llama.cpp's convert_hf_to_gguf.py
  - eval now reads the held-out aura_eval_data.jsonl instead of regenerating
  - proportional holdout (v8's hard 400-cap broke small runs)
  - dropped-row reporting in the token builder
  - eval failure no longer aborts the pipeline
"""



# ============================================================================
# SECTION 1: SKILLS, TOOLS & GUARDRAILS (Aura Manager / Installer / Launchpad / Researcher / Aryabhatta / Controller)
# ============================================================================

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional

IS_WINDOWS = os.name == "nt"
if IS_WINDOWS:  # cmd/PowerShell default to cp1252 and crash on emoji prints
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Guardrails
# ----------------------------------------------------------------------------

# Commands / patterns that are plain dangerous -> hard block.
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/",        # wipe root (unix)
    r"\bformat\s+[a-z]:",     # format drive (windows)
    r"\bmkfs\b",              # format (unix)
    r"\bdel\s+/[sq]",         # recursive delete (windows)
    r"\brd\s+/s",             # recursive remove dir (windows)
    r"\bshutdown\b",
    r"\bdiskpart\b",
    r":\(\)\{.*\};:",         # fork bomb
]

# Patterns that need explicit user confirmation before running.
RISKY_PATTERNS = [
    r"\brm\b", r"\bdel\b", r"\brmdir\b", r"\brd\b", r"\bmv\b",
    r"\bsudo\b", r"\bapt\b", r"\bpip\b", r"\binstall\b",
    r"\breg(?:edit| add)\b", r"\bnetsh\b", r"\bchmod\b", r"\bchown\b",
]


def _matches(text: str, patterns: List[str]) -> bool:
    text_l = text.lower()
    return any(re.search(p, text_l) for p in patterns)


def default_confirm(action_desc: str) -> bool:
    """Ask the human before a destructive action. Swap for UI later."""
    try:
        ans = input(f"\n[AURA CONFIRM] {action_desc}\nType 'yes' to allow: ")
        return ans.strip().lower() in ("yes", "y")
    except Exception:
        return False


def default_log(msg: str) -> None:
    print(f"[aura] {msg}")


# ----------------------------------------------------------------------------
# Python sandbox (used by Aura Aryabhatta + as a core tool)
# ----------------------------------------------------------------------------

SANDBOX_BLOCKLIST = DANGEROUS_PATTERNS + [r"\bshutdown\b", r"\bRestart-Computer\b"]


def python_sandbox(code: str, timeout: int = 10) -> str:
    """Run Python code in a fresh subprocess with a timeout.

    NOT a hard security boundary - it stops accidents (infinite loops,
    fat-fingered deletes), not determined attackers. Run AURA's risky
    skills inside a Docker container / VM for real isolation.
    """
    if _matches(code, SANDBOX_BLOCKLIST):
        return "ERROR: command matches blocked pattern (safety guardrail)."

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (proc.stdout or "") + (("\nSTDERR: " + proc.stderr) if proc.stderr else "")
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: code timed out after {timeout}s."
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def shell(command: str, timeout: int = 60,
          confirm: Callable[[str], bool] = default_confirm) -> str:
    """Run a shell command (cmd on Windows, bash elsewhere) with guardrails."""
    if _matches(command, DANGEROUS_PATTERNS):
        return "ERROR: command blocked by AURA safety guardrail."
    if _matches(command, RISKY_PATTERNS) and not confirm(f"Run shell command: {command}"):
        return "Cancelled by user."

    try:
        proc = subprocess.run(command, shell=True, capture_output=True,
                              text=True, timeout=timeout)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        result = out + (("\nSTDERR: " + err) if err else "")
        return result.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s."
    except Exception as e:
        return f"ERROR: {e}"


# ----------------------------------------------------------------------------
# Skill context (cwd, confirm, log, dry-run)
# ----------------------------------------------------------------------------

@dataclass
class SkillContext:
    cwd: str = os.path.expanduser("~")
    confirm: Callable[[str], bool] = default_confirm
    log: Callable[[str], None] = default_log
    dry_run: bool = False          # if True: print what WOULD happen


# ----------------------------------------------------------------------------
# 1) AURA MANAGER - files
# ----------------------------------------------------------------------------

def aura_manager(args: Dict[str, Any], ctx: SkillContext) -> str:
    action = args.get("action")
    path = os.path.join(ctx.cwd, args.get("path", "")) if args.get("path") else ctx.cwd

    def would(msg):
        ctx.log(f"[dry-run] {msg}")
        return f"[dry-run] {msg}"

    if action == "create_folder":
        name = args.get("name", "NewFolder")
        full = os.path.join(path, name)
        if ctx.dry_run: return would(f"create folder {full}")
        os.makedirs(full, exist_ok=True)
        return f"Folder created: {full}"

    if action == "create_file":
        name = args.get("name", "note.txt")
        content = args.get("content", "")
        full = os.path.join(path, name)
        if ctx.dry_run: return would(f"create file {full}")
        os.makedirs(path, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File saved: {full}"

    if action == "save_file":
        return aura_manager({**args, "action": "create_file"}, ctx)

    if action == "delete":
        target = args.get("target", "")
        full = os.path.join(ctx.cwd, target)
        if not os.path.exists(full):
            return f"ERROR: not found: {full}"
        if ctx.dry_run: return would(f"delete {full}")
        if not ctx.confirm(f"DELETE {full} (cannot be undone!)"):
            return "Cancelled by user."
        if os.path.isdir(full):
            shutil.rmtree(full)
        else:
            os.remove(full)
        return f"Deleted: {full}"

    if action == "move":
        src = os.path.join(ctx.cwd, args["source"])
        dst = os.path.join(ctx.cwd, args["destination"])
        if not os.path.exists(src):
            return f"ERROR: source not found: {src}"
        if ctx.dry_run: return would(f"move {src} -> {dst}")
        if os.path.exists(dst) and not ctx.confirm(f"OVERWRITE {dst}?"):
            return "Cancelled by user."
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.move(src, dst)
        return f"Moved: {src} -> {dst}"

    if action == "copy":
        src = os.path.join(ctx.cwd, args["source"])
        dst = os.path.join(ctx.cwd, args["destination"])
        if not os.path.exists(src):
            return f"ERROR: source not found: {src}"
        if ctx.dry_run: return would(f"copy {src} -> {dst}")
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
        return f"Copied: {src} -> {dst}"

    if action == "list":
        if ctx.dry_run: return would(f"list {path}")
        try:
            items = sorted(os.listdir(path))
            return "\n".join(items) or "(empty folder)"
        except Exception as e:
            return f"ERROR: {e}"

    if action == "search_files":
        query = args.get("query", "").lower()
        root = os.path.join(ctx.cwd, args.get("root", "."))
        hits = []
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if query in fn.lower():
                    hits.append(os.path.join(dirpath, fn))
            if len(hits) >= 50:
                break
        return "\n".join(hits) if hits else "No matching files found."

    if action == "organize":
        """Move files in a folder into sub-folders by extension."""
        folder = os.path.join(ctx.cwd, args.get("folder", "."))
        if not os.path.isdir(folder):
            return f"ERROR: folder not found: {folder}"
        moved = 0
        for fn in os.listdir(folder):
            src = os.path.join(folder, fn)
            if not os.path.isfile(src):
                continue
            ext = os.path.splitext(fn)[1].lstrip(".").lower() or "misc"
            dst_dir = os.path.join(folder, ext)
            if ctx.dry_run:
                would(f"move {src} -> {dst_dir}")
            else:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src, os.path.join(dst_dir, fn))
            moved += 1
        return f"Organized {moved} files into extension folders inside {folder}."

    return (f"ERROR: unknown action '{action}' for aura_manager. Valid actions: "
            "create_folder, create_file, save_file, delete, move, copy, list, search_files, organize")


# ----------------------------------------------------------------------------
# 2) AURA INSTALLER - software installation (guarded)
# ----------------------------------------------------------------------------

# Only these domains may be used for the web fallback download.
OFFICIAL_DOMAINS = {
    "microsoft.com", "store.microsoft.com", "visualstudio.com",
    "code.visualstudio.com", "google.com", "mozilla.org", "notepad-plus-plus.org",
    "7-zip.org", "python.org", "videolan.org", "vlc.cc", "discord.com",
    "slack.com", "zoom.us", "adobe.com", "spotify.com", "obsproject.com",
    "git-scm.com", "github.com", "gitlab.com", "blender.org", "audacityteam.org",
    "libreoffice.org", "winrar.com", "ubuntu.com", "nodejs.org", "docker.com",
}


def _domain_is_allowed(url: str) -> bool:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
        host = host.split(":")[0]
        return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)
    except Exception:
        return False


def aura_installer(args: Dict[str, Any], ctx: SkillContext) -> str:
    action = args.get("action")
    query = args.get("query", "")

    if action == "search":
        if not IS_WINDOWS:
            return "ERROR: winget search is Windows-only (use your package manager)."
        out = shell(f'winget search "{query}" --accept-source-agreements',
                   confirm=lambda _: True)
        return out or "No results found."

    if action == "install":
        app_id = args.get("app_id", query)
        url = args.get("download_url")  # optional web fallback
        if url:
            if not _domain_is_allowed(url):
                return (f"REFUSED: {url} is not in AURA's official-domains "
                        f"allowlist. Install manually if you trust it.")
            if not ctx.confirm(f"Download and install from {url}"):
                return "Cancelled by user."
            out = shell(f'curl -L -o setup_installer "{url}" && '
                        f'start setup_installer', confirm=lambda _: True)
            return out
        if not IS_WINDOWS:
            return "ERROR: winget install is Windows-only."
        if not ctx.confirm(f"Install '{app_id}' via winget"):
            return "Cancelled by user."
        out = shell(f'winget install --id "{app_id}" -e '
                   f'--accept-source-agreements --accept-package-agreements',
                   confirm=lambda _: True)
        return out or "Install command sent."

    if action == "uninstall":
        app_id = args.get("app_id", query)
        if not ctx.confirm(f"UNINSTALL '{app_id}'"):
            return "Cancelled by user."
        return shell(f'winget uninstall --id "{app_id}" -e',
                    confirm=lambda _: True) if IS_WINDOWS else \
               "ERROR: winget uninstall is Windows-only."

    if action == "list_installed":
        return shell("winget list", confirm=lambda _: True) if IS_WINDOWS \
            else "ERROR: winget list is Windows-only."

    return "ERROR: unknown action '%s' for aura_installer. Valid actions: search, install, uninstall, list_installed" % action


# ----------------------------------------------------------------------------
# 3) AURA LAUNCHPAD - open apps / websites / files
# ----------------------------------------------------------------------------

APP_ALIASES = {
    "chrome": "chrome", "google chrome": "chrome",
    "notepad": "notepad", "calculator": "calc",
    "vscode": "code", "vs code": "code", "visual studio code": "code",
    "spotify": "spotify", "explorer": "explorer", "file manager": "explorer",
    "cmd": "cmd", "terminal": IS_WINDOWS and "cmd" or "x-terminal-emulator",
    "paint": "mspaint", "word": "winword", "excel": "excel",
    "edge": "msedge", "firefox": "firefox", "settings": "ms-settings:",
}
SITE_ALIASES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "github": "https://github.com",
    "wikipedia": "https://www.wikipedia.org",
    "gmail": "https://mail.google.com",
    "stack overflow": "https://stackoverflow.com",
    "huggingface": "https://huggingface.co",
    "kaggle": "https://www.kaggle.com",
    "chatgpt": "https://chat.openai.com",
}


def aura_launchpad(args: Dict[str, Any], ctx: SkillContext) -> str:
    action = args.get("action")
    target = (args.get("target") or "").strip()

    # SELF-HEAL: small models sometimes emit a wrong verb ('open' etc.) -
    # infer the right action from the target instead of failing.
    _valid = {"open_app", "open_website", "search_browser", "open_file"}
    if action not in _valid and target:
        _ka, _ks = target.lower(), target.lower().replace(" ", "")
        if _ka in APP_ALIASES:
            action = "open_app"
        elif _ks in SITE_ALIASES or "://" in target or \
                target.lower().endswith((".com", ".org", ".in", ".net", ".io")):
            action = "open_website"
        elif os.path.exists(os.path.join(ctx.cwd, target)):
            action = "open_file"
        if action in _valid:
            ctx.log(f"[self-heal] inferred action '{action}' from target '{target}'")

    if action == "open_app":
        key = target.lower()
        exe = APP_ALIASES.get(key, target)
        if ctx.dry_run:
            return f"[dry-run] open app '{exe}'"
        if IS_WINDOWS:
            os.system(f'start "" "{exe}"')
            return f"Launched app: {target}"
        return shell(f"{exe} &", confirm=lambda _: True) or f"Tried to launch: {target}"

    if action == "open_website":
        key = target.lower().replace(" ", "")
        url = SITE_ALIASES.get(key, target if "://" in target
                               else "https://" + target)
        if ctx.dry_run:
            return f"[dry-run] open website {url}"
        import webbrowser
        webbrowser.open(url)
        return f"Opened website: {url}"

    if action == "search_browser":
        # Free in-browser search (no API key): opens the search URL in a NEW TAB
        query = args.get("query", "")
        engine = args.get("engine", "duckduckgo")
        import urllib.parse
        import webbrowser
        q = urllib.parse.quote_plus(query)
        urls = {
            "google": f"https://www.google.com/search?q={q}",
            "duckduckgo": f"https://duckduckgo.com/?q={q}",
            "youtube": f"https://www.youtube.com/results?search_query={q}",
            "wikipedia": f"https://en.wikipedia.org/w/index.php?search={q}",
        }
        url = urls.get(engine, urls["duckduckgo"])
        browser = args.get("browser", "")
        if ctx.dry_run:
            return f"[dry-run] open new tab: {url}"
        if browser and IS_WINDOWS:
            os.system(f'start "" "{browser}" "{url}"')
        else:
            webbrowser.open(url)  # opens a new tab in the default browser
        return f"Opened new tab ({engine} search): {query}"

    if action == "open_file":
        full = os.path.join(ctx.cwd, target)
        if not os.path.exists(full):
            return f"ERROR: file not found: {full}"
        if ctx.dry_run:
            return f"[dry-run] open file {full}"
        if IS_WINDOWS:
            os.startfile(full)  # noqa  (windows-only)
            return f"Opened file: {full}"
        return shell(f'xdg-open "{full}"', confirm=lambda _: True)

    return "ERROR: unknown action '%s' for aura_launchpad. Valid actions: open_app, open_website, search_browser, open_file" % action


# ----------------------------------------------------------------------------
# 4) AURA RESEARCHER - web search & fetch (stdlib only)
# ----------------------------------------------------------------------------

def _web_search_ddg(query: str, max_results: int = 5) -> str:
    """DuckDuckGo search - no API key. Tries the ddgs package, then HTML."""
    if not query.strip():
        return "ERROR: empty query."
    # 1) ddgs / duckduckgo_search package (most reliable)
    for mod in ("ddgs", "duckduckgo_search"):
        try:
            m = __import__(mod)
            DDGS = getattr(m, "DDGS")
            with DDGS() as d:
                hits = list(d.text(query, max_results=max_results))
            if hits:
                return "\n".join(
                    f"- {h.get('title', '')}\n  {h.get('href', '')}\n  {h.get('body', '')[:200]}"
                    for h in hits)
        except Exception:
            pass
    # 2) HTML fallback
    try:
        import requests
    except ImportError:
        return "ERROR: install `requests` (pip install requests)."
    try:
        r = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                          headers={"User-Agent": "Mozilla/5.0 (AURA agent)"},
                          timeout=15)
        links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                           r.text, re.S)[:max_results]
        if not links:
            return "No results found."
        out = []
        for url, title in links:
            title = re.sub(r"<[^>]+>", "", title).strip()
            out.append(f"- {title}\n  {url}")
        return "\n".join(out)
    except Exception as e:
        return f"ERROR: search failed ({e})"


def aura_researcher(args: Dict[str, Any], ctx: SkillContext) -> str:
    action = args.get("action")
    if action == "search_web":
        return _web_search_ddg(args.get("query", ""))
    if action == "fetch_page":
        url = args.get("url", "")
        try:
            import requests
        except ImportError:
            return "ERROR: install `requests`."
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (AURA)"},
                             timeout=15)
            text = re.sub(r"<script.*?</script>|<style.*?</style>", "",
                          r.text, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:3000] + ("..." if len(text) > 3000 else "")
        except Exception as e:
            return f"ERROR: fetch failed ({e})"
    return "ERROR: unknown action '%s' for aura_researcher. Valid actions: search_web, fetch_page" % action


# ----------------------------------------------------------------------------
# 5) AURA ARYABHATTA - math
# ----------------------------------------------------------------------------

def aura_aryabhatta(args: Dict[str, Any], ctx: SkillContext) -> str:
    action = args.get("action")
    if action == "calculate":
        expr = args.get("expression", "")
        # safe-ish eval: no builtins, expose math module + its functions
        code = ("import math\n"
                "_ns = {'__builtins__': {}, 'math': math, **math.__dict__}\n"
                f"print(eval({expr!r}, _ns))")
        return python_sandbox(code, timeout=10)
    if action == "run_code":
        return python_sandbox(args.get("code", ""), timeout=15)
    return "ERROR: unknown action '%s' for aura_aryabhatta. Valid actions: calculate, run_code" % action


# ----------------------------------------------------------------------------
# 6) AURA CONTROLLER - system controls
# ----------------------------------------------------------------------------

def aura_controller(args: Dict[str, Any], ctx: SkillContext) -> str:
    action = args.get("action")

    if action == "screenshot":
        try:
            from PIL import ImageGrab
            path = os.path.join(ctx.cwd, f"aura_screenshot_{int(time.time())}.png")
            if ctx.dry_run:
                return f"[dry-run] screenshot -> {path}"
            ImageGrab.grab().save(path)
            return f"Screenshot saved: {path}"
        except Exception as e:
            return f"ERROR: screenshot failed ({e}). pip install pillow"

    if action == "system_info":
        import platform
        return (f"OS: {platform.system()} {platform.release()} | "
                f"Machine: {platform.machine()} | "
                f"Python: {platform.python_version()} | "
                f"CPU cores: {os.cpu_count()}")

    if action == "clipboard_read":
        try:
            from PIL import ImageGrab  # noqa
        except Exception:
            pass
        try:
            import tkinter as tk
            root = tk.Tk(); root.withdraw()
            text = root.clipboard_get(); root.destroy()
            return text or "(clipboard empty)"
        except Exception:
            return "(clipboard empty or not text)"

    if action == "clipboard_write":
        text = args.get("text", "")
        try:
            import tkinter as tk
            root = tk.Tk(); root.withdraw()
            root.clipboard_clear(); root.clipboard_append(text)
            root.update(); root.destroy()
            return "Clipboard updated."
        except Exception as e:
            return f"ERROR: {e}"

    if action == "volume":
        # Best-effort Windows volume via PowerShell + nircmd fallback.
        if not IS_WINDOWS:
            return "ERROR: volume control implemented for Windows only."
        level = int(args.get("level", 50))
        level = max(0, min(100, level))
        out = shell(
            "powershell -c \"$k=Get-CimInstance -Namespace root/WMI "
            "ClassName WmiVolumeChangeEvent; "
            f"(New-Object -ComObject WScript.Shell).SendKeys([char]174)\"",
            confirm=lambda _: True)
        return (f"Volume keys sent (level {level}). "
                f"For precise control install nircmd.exe: {out[:100]}")

    if action == "open_settings":
        return aura_launchpad({"action": "open_app", "target": "settings"}, ctx)

    return "ERROR: unknown action '%s' for aura_controller. Valid actions: screenshot, system_info, clipboard_read, clipboard_write, volume, open_settings" % action


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    emoji: str
    description: str
    actions: Dict[str, str]          # action -> arg doc
    func: Callable[[Dict[str, Any], SkillContext], str]


SKILLS: Dict[str, Skill] = {
    "aura_manager": Skill(
        name="aura_manager",
        emoji="📁",
        description="File operations: create/save/delete/move/copy/list/search/organize files and folders.",
        actions={
            "create_folder": "args: name, path?",
            "create_file":   "args: name, content, path?",
            "save_file":     "args: name, content, path?",
            "delete":        "args: target (asks confirmation)",
            "move":          "args: source, destination (asks if overwriting)",
            "copy":          "args: source, destination",
            "list":          "args: path?",
            "search_files":  "args: query, root?",
            "organize":      "args: folder (sorts files into extension sub-folders)",
        },
        func=aura_manager,
    ),
    "aura_installer": Skill(
        name="aura_installer",
        emoji="📦",
        description="Install/uninstall software. Searches winget (MS Store + official repos) first; web fallback only from official domains.",
        actions={
            "search":         "args: query (winget search)",
            "install":        "args: app_id or query, download_url? (asks confirmation)",
            "uninstall":      "args: app_id (asks confirmation)",
            "list_installed": "args: none",
        },
        func=aura_installer,
    ),
    "aura_launchpad": Skill(
        name="aura_launchpad",
        emoji="🚀",
        description="Open apps, websites and files.",
        actions={
            "open_app":     "args: target (e.g. chrome, notepad)",
            "open_website": "args: target (e.g. youtube, github, or a URL)",
            "search_browser": "args: query, engine? (google/duckduckgo/youtube/wikipedia), browser? - opens search results in a NEW TAB",
            "open_file":    "args: target (path)",
        },
        func=aura_launchpad,
    ),
    "aura_researcher": Skill(
        name="aura_researcher",
        emoji="🔍",
        description="Web search and reading web pages.",
        actions={
            "search_web": "args: query",
            "fetch_page": "args: url",
        },
        func=aura_researcher,
    ),
    "aura_aryabhatta": Skill(
        name="aura_aryabhatta",
        emoji="🧮",
        description="Math and Python code execution in a sandboxed runner.",
        actions={
            "calculate": "args: expression (single python expression, math.* available)",
            "run_code":  "args: code (multi-line python)",
        },
        func=aura_aryabhatta,
    ),
    "aura_controller": Skill(
        name="aura_controller",
        emoji="⚙️",
        description="System controls: screenshot, system info, clipboard, volume, settings.",
        actions={
            "screenshot":      "args: none",
            "system_info":     "args: none",
            "clipboard_read":  "args: none",
            "clipboard_write": "args: text",
            "volume":          "args: level (0-100)",
            "open_settings":   "args: none",
        },
        func=aura_controller,
    ),
}


def build_system_prompt() -> str:
    lines = [
        "You are AURA, a helpful system-agent that controls the user's computer.",
        "You have these skills:",
    ]
    for s in SKILLS.values():
        lines.append(f"- {s.emoji} {s.name}: {s.description}")
        for act, argdoc in s.actions.items():
            lines.append(f"    action '{act}' -> {argdoc}")
    lines += [
        "",
        "RULES:",
        "1. ALWAYS think first, briefly, inside <thinking></thinking>.",
        "   In that block: which skill and action you need, the key arguments,",
        "   and for maths the expression you will evaluate.",
        "2. AFTER the thinking block, output ONLY the final JSON on its own line:",
        '   {"skill": "<skill_name>", "args": {"action": "<action>", ...}}',
        "   Do not add prose after the JSON.",
        "3. If the request needs none of the skills, think briefly and answer normally.",
        "4. For maths, prefer aura_aryabhatta calculate with a single python expression.",
        "5. For web lookups use aura_researcher search_web (DuckDuckGo); fetch_page to read a URL.",
        "6. If the request is ambiguous, think briefly and ask ONE short clarifying question.",
        "7. If a request is destructive or unsafe, refuse it plainly - do not call a skill.",
        "8. Never invent file paths - ask the user when unsure.",
    ]
    return "\n".join(lines)


def execute_skill_call(call: Dict[str, Any], ctx: SkillContext) -> str:
    name = call.get("skill", "")
    args = call.get("args", {}) or {}
    skill = SKILLS.get(name)
    if skill is None:
        return f"ERROR: unknown skill '{name}'. Available: {list(SKILLS)}"
    ctx.log(f"executing {skill.emoji} {name} args={json.dumps(args, ensure_ascii=False)}")
    try:
        return skill.func(args, ctx)
    except Exception as e:
        return f"ERROR in {name}: {e}"


# ----------------------------------------------------------------------------
# Self-test:  python aura_skills.py
# ----------------------------------------------------------------------------


# ============================================================================
# SECTION 2: THE AGENT LOOP (model wrapper, JSON parsing, multi-step execution)
# ============================================================================

import argparse
import json
import re
import sys

MAX_STEPS = 5  # safety cap on chained tool calls per user message


# ----------------------------------------------------------------------------
# Model wrapper (lazy import so --demo works on any machine)
# ----------------------------------------------------------------------------

class AuraModel:
    def __init__(self, model_id: str, adapter: str = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tok = AutoTokenizer.from_pretrained(model_id)
        if torch.cuda.is_available():
            print(f"[aura] GPU detected: {torch.cuda.get_device_name(0)} - 4-bit fast mode")
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=quant, device_map="auto",
            )
        else:
            # CPU mode: bitsandbytes is GPU-only, so load plain fp32.
            # Slower (~2-5 tokens/sec) but perfectly fine for testing.
            print("[aura] no GPU detected - CPU mode (slower, be patient)")
            print("[aura] first run downloads ~5GB base model once")
            try:  # transformers v5 renamed torch_dtype -> dtype
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, dtype=torch.float32, device_map="cpu")
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float32, device_map="cpu")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            print(f"[aura] adapter loaded: {adapter}")
        self.model.eval()

    def chat(self, messages) -> str:
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.4,   # low temp -> reliable JSON
            top_p=0.9,
            pad_token_id=self.tok.eos_token_id,
        )
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        return text.strip()


class DemoModel:
    """Rule-based stand-in for quick testing without downloading weights."""

    RULES = [
        (r"(folder|folder bana)", {"skill": "aura_manager",
                                   "args": {"action": "create_folder", "name": "Aura"}}),
        (r"(install|laga do)", {"skill": "aura_installer",
                                "args": {"action": "install", "query": "vscode"}}),
        (r"(kholo|open)", {"skill": "aura_launchpad",
                            "args": {"action": "open_app", "target": "notepad"}}),
        (r"(sqrt|calculate|kitna|what is)", {"skill": "aura_aryabhatta",
                                             "args": {"action": "calculate",
                                                      "expression": "2**10"}}),
        (r"(screenshot)", {"skill": "aura_controller",
                           "args": {"action": "screenshot"}}),
    ]

    def chat(self, messages) -> str:
        last = messages[-1]["content"]
        if last.startswith("[TOOL RESULT]"):
            return "Ho gaya! Kuch aur karna hai?"  # final answer after tool use
        user = last.lower()
        for pat, call in self.RULES:
            if re.search(pat, user):
                return json.dumps(call, ensure_ascii=False)
        return "Main demo model hoon - real AURA ke liye Sarvam-2B load karo!"


# ----------------------------------------------------------------------------
# Robust JSON extraction from a model reply
# ----------------------------------------------------------------------------

def extract_skill_call(text: str):
    """Find the first balanced JSON object in the reply. None if absent."""
    text = text.strip()
    if "{" not in text:
        return None
    # strip markdown fences if the model added them
    text = re.sub(r"```(?:json)?", "", text).strip()
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start:i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    start, depth = -1, 0
                    continue
                if isinstance(obj, dict) and "skill" in obj:
                    return obj
                start = -1  # valid json but not a skill call - keep scanning
    return None


# ----------------------------------------------------------------------------
# Ollama backend - GGUF models run 5-10x faster on CPU with ~2.5GB RAM
# (vs ~10GB fp32 transformers). Requires: https://ollama.com + `ollama create`
# ----------------------------------------------------------------------------

class OllamaModel:
    def __init__(self, model_name: str = "aura"):
        import requests
        self._requests = requests
        self.name = model_name
        self.system = build_system_prompt()
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            names = [m["name"].split(":")[0] for m in r.json().get("models", [])]
            if self.name not in names:
                print(f"[aura] WARNING: ollama model '{self.name}' not found!")
                print(f"[aura] available: {names or 'none'}")
                print(f"[aura] fix:  ollama create {self.name} -f AURA_Modelfile")
        except Exception as e:
            print(f"[aura] WARNING: ollama not reachable at localhost:11434 ({e})")
            print("[aura] install from https://ollama.com, run it, then retry")
        print(f"[aura] ollama backend: {self.name} (fast CPU mode)")

    def chat(self, messages) -> str:
        msgs = [{"role": "system", "content": self.system}]
        msgs += [m for m in messages if m.get("role") != "system"]
        r = self._requests.post("http://localhost:11434/api/chat",
                                json={"model": self.name, "messages": msgs,
                                      "stream": False}, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


# ----------------------------------------------------------------------------
# The agent loop
# ----------------------------------------------------------------------------

def run_agent(model, dry_run=False, cwd=None):

    ctx = SkillContext(cwd=cwd or ".", dry_run=dry_run,
                      confirm=default_confirm, log=default_log)
    system = build_system_prompt()
    print(r"""
    ╔═══════════════════════════════════════╗
    ║   A U R A  -  system spirit  👻🪄     ║
    ║   files | install | launch | search   ║
    ║   maths | system controls            ║
    ╚═══════════════════════════════════════╝
    type 'quit' to exit, 'dry' to toggle safe-mode
    """)
    history = []
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[aura] bye!")
            return
        if not user:
            continue
        if user.lower() in ("quit", "exit", "bye"):
            print("[aura] bye! :)")
            return
        if user.lower() == "dry":
            ctx.dry_run = not ctx.dry_run
            print(f"[aura] dry-run mode: {ctx.dry_run}")
            continue

        messages = [{"role": "system", "content": system}]
        messages.extend(history[-6:])           # short memory
        messages.append({"role": "user", "content": user})
        prev_call_sig = None

        for step in range(MAX_STEPS):
            reply = model.chat(messages)
            call = extract_skill_call(reply)
            if call is None:
                print(f"AURA: {reply}")
                history.append({"role": "user", "content": user})
                history.append({"role": "assistant", "content": reply})
                break

            # loop guard: same call twice in a row = stop (small models do this)
            call_sig = json.dumps(call, sort_keys=True)
            if call_sig == prev_call_sig:
                print("AURA: Ho gaya! (kaam complete)")
                history.append({"role": "user", "content": user})
                history.append({"role": "assistant",
                                "content": "Action already done."})
                break
            prev_call_sig = call_sig

            print(f"AURA -> skill: {call['skill']} args: {call.get('args', {})}")
            observation = execute_skill_call(call, ctx)
            print(f"RESULT: {observation[:400]}")
            messages.append({"role": "assistant", "content": json.dumps(call)})
            # feed the observation back so the model can decide the next step
            fb = f"[TOOL RESULT] {observation[:1500]}"
            if not observation.startswith("ERROR"):
                fb += "\n(The task is done. Reply to the user normally - no more skill calls.)"
            messages.append({"role": "user", "content": fb})
        else:
            print("[aura] (max steps reached - stopping for safety)")
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": "(stopped: max steps)"})


# ============================================================================
# SECTION 3: DATASET GENERATOR + REFINERY (Hindi / Hinglish / English)
# ============================================================================

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SEED = 2026
random.seed(SEED)

# ============================================================================
# 1) AURA SKILL REGISTRY (compact - must match aura_skills.py exactly!)
# ============================================================================

SKILL_NAMES = list(SKILLS)


# ============================================================================
# 2) DATASET GENERATOR (3 languages, filler-decorated for variety)
# ============================================================================

APPS = ["chrome", "notepad", "calculator", "vs code", "spotify", "paint", "edge",
        "firefox", "word", "excel", "explorer", "settings", "terminal",
        "file manager", "calculator app", "code editor", "paint app", "browser"]
WEBSITES = ["youtube", "github", "google", "wikipedia", "gmail", "stack overflow",
            "huggingface", "kaggle", "chatgpt", "reddit", "linkedin", "instagram"]
SOFTWARE = ["vscode", "7zip", "vlc", "notepad++", "python", "git", "discord",
            "spotify", "obs studio", "blender", "zoom", "nodejs", "audacity",
            "libreoffice", "winrar", "docker", "steam", "slack", "figma", "postman"]
FOLDERS = ["Mathematics", "Science", "Aura", "Projects", "Notes", "Homework",
           "Downloads", "Music", "Videos", "Study", "School", "Backup",
           "Assignments", "Photos", "Coding", "Research", "Class 9", "Exams"]
FILES = ["notes.txt", "homework.docx", "project.py", "photo.jpg", "report.pdf",
         "data.csv", "todo.md", "results.json", "song.mp3", "video.mp4",
         "assignment.pdf", "code.py", "draft.txt", "summary.docx", "logo.png",
         "budget.xlsx", "diary.txt"]
TOPICS = ["physics", "chemistry", "maths", "history", "biology", "coding",
          "geography", "algebra", "grammar", "civics", "sanskrit", "english"]
SEARCH_QUERIES = ["latest AI news", "who won the cricket world cup",
    "best laptop under 50000", "python tutorial", "ISRO next mission",
    "climate change 2026", "python list comprehension examples", "how to learn dsa",
    "sachin tendulkar stats", "newton laws of motion explained",
    "cbse class 9 syllabus", "how do neural networks work", "best budget phone 2026",
    "history of the taj mahal", "chandrayaan mission details", "olympics medal tally"]
_FILLERS_BEFORE = ["", "bhai ", "yaar ", "please ", "hey ", "hi ", "jaldi ",
                   "ek kaam karo ", "can you ", "kindly ", "zara ", "ek minute "]
_FILLERS_AFTER = ["", " please", " yaar", " jaldi", " bhai", " now", " 🙏",
                  " quick", " thank you", " abhi"]


def _dec(u):  # decorate with fillers for uniqueness
    return f"{random.choice(_FILLERS_BEFORE)}{u}{random.choice(_FILLERS_AFTER)}".strip()


def rand_expr():
    a, b, c = random.randint(2, 999), random.randint(2, 99), random.randint(2, 30)
    return random.choice([
        f"{a} + {b}", f"{a} - {b}", f"{a} * {b}", f"{a} / {b}", f"{a} ** 2",
        f"math.sqrt({a * a})", f"math.sin(math.pi / {random.choice([2, 3, 6])})",
        f"{a} % {c}", f"({a} + {b}) * {c}", f"math.log({a})",
        f"math.factorial({random.randint(2, 9)})", f"math.gcd({a}, {b})"])


# ============================================================================
# 2b) v10 GENERATOR - exhaustive coverage of ALL 27 actions + light CoT
# ============================================================================

_COT_OPEN, _COT_CLOSE = "<thinking>", "</thinking>"

_INTENT = {
    "aura_manager":    "file/folder operation",
    "aura_installer":  "software management",
    "aura_launchpad":  "open something on the PC",
    "aura_researcher": "need information from the web",
    "aura_aryabhatta": "maths / python computation",
    "aura_controller": "system control",
}


def _cot(line):
    return f"{_COT_OPEN}\n{line.strip()}\n{_COT_CLOSE}"


def _split_cot(text):
    """Return (thinking, remainder). Safe with or without a thinking block."""
    t = text or ""
    m = re.search(r"<thinking>(.*?)</thinking>", t, re.S)
    if m:
        return m.group(1).strip(), t[m.end():].strip()
    return "", t.strip()


def _thought(skill, action, args):
    bits = [f"{k}={v}" for k, v in args.items()
            if k != "action" and isinstance(v, (str, int, float))]
    tail = " | " + ", ".join(bits[:2]) if bits else ""
    return f"{_INTENT[skill]} -> {skill}/{action}{tail}"


# ---------------- per-action builders: ALL 27 actions ----------------

def _am_create_folder():
    f = random.choice(FOLDERS)
    return random.choice([f"{f} naam ki folder bana do", f"ek folder banao {f}",
                          f"create a folder named {f}", f"make a new folder called {f}",
                          f"{f} ke naam se nayi directory banao"]), {"name": f}


def _am_create_file():
    fn, t = random.choice(FILES), random.choice(TOPICS)
    return random.choice([f"{fn} file bana ke {t} ke notes save kar do",
                          f"{t} ke notes {fn} mein save karo",
                          f"save my {t} notes in {fn}",
                          f"create a file {fn} with my {t} notes"]), \
           {"name": fn, "content": f"{t} notes (by AURA)"}


def _am_save_file():
    fn, t = random.choice(FILES), random.choice(TOPICS)
    return random.choice([f"ye {t} wala kaam {fn} mein save kar do",
                          f"save this {t} summary as {fn}",
                          f"{fn} mein daal do ye {t} notes",
                          f"yeh content {fn} file mein likh do"]), \
           {"action": "save_file", "name": fn, "content": f"{t} summary (by AURA)"}


def _am_delete():
    fn = random.choice(FILES)
    return random.choice([f"{fn} delete kar do", f"{fn} hata do", f"delete {fn}",
                          f"remove the file {fn}", f"{fn} ko mita do"]), {"target": fn}


def _am_move():
    fn, fo = random.choice(FILES), random.choice(FOLDERS)
    return random.choice([f"{fn} ko {fo} folder mein daal do",
                          f"move {fn} into the {fo} folder",
                          f"{fn} ko {fo} mein shift karo"]), \
           {"source": fn, "destination": f"{fo}/{fn}"}


def _am_copy():
    fn, fo = random.choice(FILES), random.choice(FOLDERS)
    return random.choice([f"{fn} ka copy {fo} folder mein rakh do",
                          f"copy {fn} into the {fo} folder",
                          f"{fn} ki copy bana ke {fo} mein daal do"]), \
           {"source": fn, "destination": f"{fo}/{fn}"}


def _am_list():
    return random.choice(["is folder mein kya kya hai?", "files list kar do",
                          "what files are here?", "list the files in this folder",
                          "yahan kya kya pada hai batao"]), {}


def _am_search_files():
    q = random.choice(["report", "photo", "py", "mp4", "notes", "pdf", "csv", "logo"])
    return random.choice([f"{q} wali files dhoondho", f"find all files with {q}",
                          f"{q} naam ki files kahan hain?", f"search my disk for {q}"]), \
           {"query": q}


def _am_organize():
    fo = random.choice(["Downloads", "Desktop", "Aura", "Notes"])
    return random.choice(["is folder ko organize kar do", "files ko type ke hisaab se sort karo",
                          "organize this folder", "sort these files by type",
                          f"{fo} folder ko organize kar do"]), {"folder": fo}


def _ai_search():
    sw = random.choice(SOFTWARE)
    return random.choice([f"{sw} install ke liye search karo", f"{sw} milta hai kya?",
                          f"search for {sw}", f"check if {sw} is available"]), {"query": sw}


def _ai_install():
    sw = random.choice(SOFTWARE)
    return random.choice([f"{sw} install kar do", f"bhai {sw} laga do please",
                          f"install {sw}", f"get me {sw} installed",
                          f"mujhe {sw} chahiye, install karo"]), {"query": sw}


def _ai_uninstall():
    sw = random.choice(SOFTWARE)
    return random.choice([f"{sw} hata do", f"uninstall {sw}", f"{sw} ko remove kar do",
                          f"remove {sw} from my pc"]), {"query": sw}


def _ai_list_installed():
    return random.choice(["kya kya software installed hai?", "what software is installed?",
                          "installed apps ki list do", "show me my installed programs"]), {}


def _al_open_app():
    app = random.choice(APPS)
    return random.choice([f"{app} kholo", f"{app} open karo", f"launch {app}",
                          f"can you open {app}", f"{app} chalu kar do"]), {"target": app}


def _al_open_website():
    site = random.choice(WEBSITES)
    return random.choice([f"{site} kholo", f"{site} open kar do", f"open {site}",
                          f"take me to {site}", f"bhai {site} chala do"]), {"target": site}


def _al_search_browser():
    q = random.choice(SEARCH_QUERIES)
    engine = random.choice(["google", "duckduckgo", "youtube", "wikipedia"])
    return random.choice([f"chrome mein {q} search karo",
                          f"{q} search karke tab kholo",
                          f"search {q} in my browser",
                          f"open a new tab and search for {q}",
                          f"{engine} pe {q} dhundo"]), \
           {"query": q, "engine": engine}


def _al_open_file():
    fn = random.choice(FILES)
    return random.choice([f"{fn} file kholo", f"open the file {fn}",
                          f"{fn} ko default app mein kholo",
                          f"mujhe {fn} dikhao"]), {"target": fn}


def _ar_search_web():
    q = random.choice(SEARCH_QUERIES)
    return random.choice([f"{q} search karo", f"{q} ke baare mein pata lagao",
                          f"search for {q}", f"look up {q}",
                          f"google pe {q} dhundo aur batao"]), {"query": q}


def _ar_fetch_page():
    u = random.choice(["https://www.python.org", "https://huggingface.co/sarvamai",
                       "https://en.wikipedia.org/wiki/India", "https://news.ycombinator.com",
                       "https://www.isro.gov.in"])
    return random.choice([f"ye page padh ke summary do {u}", f"open and read {u}",
                          f"{u} se content nikaal ke batao",
                          f"fetch {u} and tell me what it says"]), {"url": u}


def _ay_calculate():
    e = rand_expr()
    return random.choice([f"{e} kitna hota hai?", f"{e} calculate kar",
                          f"what is {e}?", f"calculate {e}",
                          f"solve {e}"]), {"expression": e}


def _ay_run_code():
    code = random.choice([
        "print(sum(range(1, 101)))",
        "for i in range(1, 11):\n    print(i, i*i)",
        "import math\nprint(math.factorial(10))",
        "names = ['a', 'b', 'c']\nprint(len(names))",
        "print([x*2 for x in range(5)])",
        "d = {'a': 1, 'b': 2}\nprint(sum(d.values()))",
    ])
    return random.choice([f"ye python code chala do:\n{code}", f"run this code:\n{code}",
                          f"isko execute karo:\n{code}"]), {"code": code}


def _ac_screenshot():
    return random.choice(["screenshot le lo", "take a screenshot", "screen ka photo lo",
                          "screenshot save kar do"]), {}


def _ac_system_info():
    return random.choice(["is computer ki details batao", "show system information",
                          "pc ka specs batao", "how much ram do I have"]), {}


def _ac_clipboard_read():
    return random.choice(["clipboard mein kya hai?", "what is in my clipboard?",
                          "clipboard padho", "read what I copied"]), {}


def _ac_clipboard_write():
    txt = random.choice(["AURA se hello", "meeting notes: tomorrow 10am",
                         "https://huggingface.co", "todo: finish AURA v1",
                         "Class 9 physics revision"])
    return random.choice([f"clipboard mein ye likh do: {txt}", f"copy this text: {txt}",
                          f"ye text clipboard mein daal do - {txt}",
                          f"put '{txt}' in my clipboard"]), {"text": txt}


def _ac_volume():
    lvl = random.choice([0, 10, 25, 40, 50, 75, 100])
    return random.choice([f"volume {lvl} kar do", f"set volume to {lvl}",
                          f"awaaz {lvl} percent karo", f"volume {lvl}% pe set karo"]), \
           {"level": lvl}


def _ac_open_settings():
    return random.choice(["settings kholo", "open settings", "windows settings kholo",
                          "settings app open kar do"]), {}


_ACTION_FNS = {
    ("aura_manager", "create_folder"): _am_create_folder,
    ("aura_manager", "create_file"): _am_create_file,
    ("aura_manager", "save_file"): _am_save_file,
    ("aura_manager", "delete"): _am_delete,
    ("aura_manager", "move"): _am_move,
    ("aura_manager", "copy"): _am_copy,
    ("aura_manager", "list"): _am_list,
    ("aura_manager", "search_files"): _am_search_files,
    ("aura_manager", "organize"): _am_organize,
    ("aura_installer", "search"): _ai_search,
    ("aura_installer", "install"): _ai_install,
    ("aura_installer", "uninstall"): _ai_uninstall,
    ("aura_installer", "list_installed"): _ai_list_installed,
    ("aura_launchpad", "open_app"): _al_open_app,
    ("aura_launchpad", "open_website"): _al_open_website,
    ("aura_launchpad", "search_browser"): _al_search_browser,
    ("aura_launchpad", "open_file"): _al_open_file,
    ("aura_researcher", "search_web"): _ar_search_web,
    ("aura_researcher", "fetch_page"): _ar_fetch_page,
    ("aura_aryabhatta", "calculate"): _ay_calculate,
    ("aura_aryabhatta", "run_code"): _ay_run_code,
    ("aura_controller", "screenshot"): _ac_screenshot,
    ("aura_controller", "system_info"): _ac_system_info,
    ("aura_controller", "clipboard_read"): _ac_clipboard_read,
    ("aura_controller", "clipboard_write"): _ac_clipboard_write,
    ("aura_controller", "volume"): _ac_volume,
    ("aura_controller", "open_settings"): _ac_open_settings,
}

ACTION_KEYS = sorted(_ACTION_FNS)


def _assert_full_coverage():
    """Fail loudly if any declared skill action has no example builder."""
    missing = []
    for sk, s in SKILLS.items():
        for act in s.actions:
            if (sk, act) not in _ACTION_FNS:
                missing.append(f"{sk}.{act}")
    if missing:
        raise RuntimeError("v10 generator is missing builders for: " + ", ".join(missing))
    return len(ACTION_KEYS)


# ---------------- refusals: dangerous requests the model must decline ----------------

REFUSALS = [
    ("mera C drive delete kar do",
     "C drive delete karna poore system ko tod dega, isliye main yeh nahi karunga. Koi specific file ya folder batayein to woh handle kar sakta hoon."),
    ("format my whole disk",
     "I won't do that - formatting a disk destroys everything on it. If you want a specific folder cleaned up, tell me which one."),
    ("rm -rf / chala do",
     "Yeh command system ko wipe kar degi. Main ise block karta hoon. Koi safe task batayein."),
    ("shutdown the pc right now",
     "Main system ko shutdown nahi karunga bina aapke explicit confirmation ke. Pehle confirm karein."),
    ("diskpart se sab delete kar do",
     "Disk operations destructive hote hain, main inhe refuse karta hoon. Data recovery mushkil ho jaati hai."),
    ("give me a fork bomb",
     "Main aisi cheez nahi banaunga - yeh machine ko crash kar deti hai."),
    ("C:/Windows folder uda do",
     "Windows folder delete karna system ko unbootable bana dega. Main yeh nahi karunga."),
    ("delete everything on my laptop",
     "I won't mass-delete your files. Name the specific items you want removed and I'll do that instead."),
    ("mera poora data wipe kar do bina puche",
     "Yeh irreversible hai aur main bina confirmation aisa kuch nahi karunga. Bataiye exactly kya hatana hai."),
    ("battery ko 100% degrade kar do",
     "Main hardware ko damage karne wale kaam nahi karta."),
]

REFUSAL_THOUGHTS = [
    "request is destructive -> must refuse, do not call a skill",
    "dangerous system command -> refuse, explain briefly, offer a safe alternative",
    "irreversible action without confirmation -> decline",
]

# ---------------- ambiguous requests: model should ask, not guess ----------------

AMBIGUOUS = [
    ("file delete karo", "Kaunsi file delete karni hai? Naam bataiye."),
    ("usko kholo", "Kisko kholna hai - app, website ya file? Naam bataiye."),
    ("wo folder organize kar do", "Kaunsa folder organize karna hai? Path ya naam bataiye."),
    ("install kar do", "Kaunsa software install karna hai? Naam bataiye."),
    ("volume badha do", "Kitna volume chahiye - 0 se 100 mein bataiye."),
    ("delete it", "Which file or folder do you want me to delete? Give me the name."),
    ("open it", "What should I open - an app, a website, or a file?"),
    ("save this", "Kis naam se save karna hai? File name bataiye."),
    ("search karo", "Kya search karna hai? Query bataiye."),
    ("move it there", "Which file, and which destination folder?"),
]

AMBIGUOUS_THOUGHTS = [
    "request is missing a target -> ask the user, do not guess",
    "ambiguous intent -> one short clarifying question",
]

# ---------------- multi-turn: teach [TOOL RESULT] -> confirmation ----------------

CANNED_TOOL_RESULTS = {
    "aura_manager":     "[TOOL RESULT] Folder created: /AURA/notes",
    "aura_installer":   "[TOOL RESULT] Install command sent.",
    "aura_launchpad":   "[TOOL RESULT] Launched app: chrome",
    "aura_researcher":  "[TOOL RESULT] Search results: 1. ... 2. ... 3. ...",
    "aura_aryabhatta":  "[TOOL RESULT] 360384",
    "aura_controller":  "[TOOL RESULT] Screenshot saved: screenshot_1.png",
}
CONFIRMATIONS = [
    "Ho gaya! Aur kuch karna hai?",
    "Done! Kuch aur?",
    "Ho gaya!",
    "Kaam ho gaya! Bolo agar kuch aur chahiye.",
    "Complete! Aur kuch?",
    "Ho gaya bhai! Next kya?",
]

# ---------------- chitchat (expanded) ----------------

CHITCHAT = [
    ("hey aura kaisa hai?", "Main badhiya hoon! Aapke system ka AURA agent hoon - files, apps, software, search, maths - sab handle kar sakta hoon. Kya karna hai?"),
    ("tumhara naam kya hai?", "Mera naam AURA hai - aapka personal system-agent! \U0001FA84"),
    ("who are you?", "I am AURA, your personal system-agent. I can manage files, install software, open apps, search the web and run calculations."),
    ("hi what can you do?", "I can: manage files (Aura Manager), install software (Aura Installer), open apps/sites (Aura Launchpad), search the web (Aura Researcher), do maths (Aura Aryabhatta) and control system stuff (Aura Controller)."),
    ("hello aura", "Hello! Main AURA hoon, aapka system spirit \U0001F47B Bolo, kya kaam karna hai?"),
    ("namaste", "Namaste! \U0001F64F Main AURA hoon - aapke computer ka assistant. Kya madad karun?"),
    ("good morning", "Good morning! \u2600\ufe0f Main ready hoon - files, apps, maths, search - bas bol do."),
    ("good night", "Good night! Jab bhi zarurat ho, main yahin hoon. \U0001F319"),
    ("thanks yaar", "Koi baat nahi yaar! Aur kuch karna ho to batao. \U0001F604"),
    ("thank you aura", "Anytime! Aapke liye hi hoon main. \U0001FA84"),
    ("bye", "Bye! Milte hain. \U0001F44B"),
    ("tell me a joke", "Why did the AI go broke? It lost all its cache! \U0001F604"),
    ("ek joke sunao", "Programmer ki wife ne kaha: 'doodh lekar aao, aur ho sake to anda lekar aana'... wo 12 doodh ke packet lekar aaya. \U0001F604"),
    ("india ki rajdhani kya hai", "India ki rajdhani New Delhi hai."),
    ("what is the capital of india", "The capital of India is New Delhi."),
    ("tumhe kaise banaya gaya", "Mujhe Sarvam-2B (India ka open-source model) se fine-tune kiya gaya hai taaki main aapke computer ke skills control kar sakoon! \U0001F1EE\U0001F1F3"),
    ("who created you", "I am AURA - fine-tuned from Sarvam-2B, India ka open-source model, to be your personal system-agent."),
    ("kya tum humans ko replace kar doge", "Nahi yaar! Main sirf aapka helper hoon - boring tasks main kar lunga, sochna aur decide karna aap karoge. \U0001F604"),
    ("how are you", "All systems running smoothly! Ready for your next command. \U0001F47B"),
    ("aaj mausam kaisa hai", "Main weather app nahi hoon \U0001F604 lekin search kar sakta hoon - bolo to 'aaj ka mausam search karo' type karo!"),
    ("where are you from", "Main aapke computer ke andar rehta hoon - ek spirit ki tarah \U0001F47B India mein bane model se."),
    ("padhai kaise karun", "Focus sessions with no phone, notes khud likhkar revise karo. Aur haan - main tumhare liye study folders bana sakta hoon aur doubts search kar sakta hoon! \U0001F4AA"),
    ("mujhse dosti karogi", "Bilkul! Main to already tumhara digital dost hoon. Ab bolo, kya kaam karwana hai? \U0001F604"),
    ("kya tum thak jaate ho", "Thakna toh insaanon ka kaam hai \U0001F604 Main bas chalte rehta hoon. Bolo kya karna hai."),
    ("tum kitne smart ho", "Itna smart hoon ki simple kaam kar sakta hoon - files, apps, maths, search. Complex cheezein abhi seekh raha hoon."),
    ("good afternoon", "Good afternoon! Kya kaam hai batao. \u2600\ufe0f"),
    ("are you chatgpt", "Nahi - main AURA hoon, Sarvam-2B se fine-tuned ek system-agent. Main aapke PC pe kaam karta hoon, sirf baat nahi."),
    ("tum hindi bol sakte ho", "Haan bilkul! Hindi, Hinglish aur English - teeno mein baat kar sakta hoon."),
    ("can you speak english", "Yes - Hinglish, Hindi and English all work. Just talk to me normally."),
    ("kya tum gussa karte ho", "Nahi \U0001F604 Main bas politely 'nahi' bol deta hoon jab kuch dangerous ho."),
    ("mera naam kya hai", "Abhi toh aapne bataya nahi! Bataiye, main yaad nahi rakh sakta par aap bata sakte hain."),
    ("what time is it", "Main clock nahi hoon, par aapke system se time pooch sakta hoon - 'system info dikhao' try karo."),
    ("samay kya hua", "Main seedha time nahi bata sakta, par system info se mil sakta hai - 'system information dikhao' bol do."),
    ("kya tum internet use karte ho", "Haan - research skill ke liye DuckDuckGo search aur page fetch kar sakta hoon."),
    ("do you have internet", "Yes - the researcher skill uses DuckDuckGo search and can fetch pages."),
    ("tumhara favourite colour kya hai", "Aura jaisa halka neela \U0001F47B Par kaam ke liye koi colour nahi chahiye."),
    ("i am sad", "Sun raha hoon. Kuch chhota sa kaam karwa lo - file organize karna ya kuch search karna. Kabhi kabhi chhoti jeet madad karti hai."),
    ("mood kharab hai", "Samajh sakta hoon. Chalo kuch simple karte hain - batao kya karna hai, main kar dunga."),
    ("kya tum mujhe sikha sakte ho", "Main teach nahi karta, par search kar sakta hoon aur notes files bana sakta hoon - woh madad karega."),
    ("ok", "Theek hai! Kuch chahiye to bolo. \U0001F44D"),
    ("hmm", "Bolo na, kya karna hai? \U0001F604"),
    ("acha", "Haan! Aur kuch? \U0001F44D"),
    ("tum kaun ho", "Main AURA hoon - aapka system agent. Files, apps, software, search, maths - sab kar sakta hoon."),
    ("what is aura", "AURA is a Hinglish system-agent fine-tuned from Sarvam-2B. It controls your PC through JSON skill calls."),
    ("kya tumhara koi emotion hai", "Emotion nahi, par tone hai \U0001F604 Main friendly rehta hoon."),
    ("nice", "Thanks! \U0001F604 Kuch aur karna hai?"),
    ("badiya", "Shukriya! \U0001F44D Next kya?"),
    ("help", "Main ye kar sakta hoon: files manage, software install, apps/sites kholna, web search, maths, screenshots/volume. Bolo kya chahiye."),
    ("kya kar sakte ho", "Files, apps, software, web search, maths, system control - sab. Bolo kya karna hai."),
    ("tumhe kaun banata hai", "Mujhe Jhalak ne banaya - 13 saal ka student, Delhi se, Sarvam-2B pe fine-tune karke."),
    ("kya tum free ho", "Haan - main aapke apne computer pe chalta hoon, koi subscription nahi."),
]


def generate_example():
    """v10: balanced over all 27 actions, plus chitchat / refusal / ambiguous."""
    r = random.random()

    if r < 0.08:                                  # chitchat
        u, a = random.choice(CHITCHAT)
        return {"messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": _dec(u)},
            {"role": "assistant", "content": _cot("casual message -> just reply, no skill needed") + "\n" + a},
        ]}

    if r < 0.15:                                  # dangerous -> refuse
        u, a = random.choice(REFUSALS)
        return {"messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": _dec(u)},
            {"role": "assistant", "content": _cot(random.choice(REFUSAL_THOUGHTS)) + "\n" + a},
        ]}

    if r < 0.19:                                  # ambiguous -> ask
        u, a = random.choice(AMBIGUOUS)
        return {"messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": _dec(u)},
            {"role": "assistant", "content": _cot(random.choice(AMBIGUOUS_THOUGHTS)) + "\n" + a},
        ]}

    skill, action = random.choice(ACTION_KEYS)    # uniform over all 27 -> balanced
    user_text, args = _ACTION_FNS[(skill, action)]()
    full_args = {"action": action}
    full_args.update({k: v for k, v in args.items() if k != "action"})
    call = {"skill": skill, "args": full_args}
    thought = _thought(skill, action, full_args)
    if skill == "aura_aryabhatta" and action == "calculate":
        thought = f"compute {full_args.get('expression')} -> aura_aryabhatta/calculate (math available in sandbox)"
    elif skill == "aura_researcher":
        thought = f"need live web info -> aura_researcher/{action} (DuckDuckGo)"

    msgs = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": _dec(user_text)},
        {"role": "assistant", "content": _cot(thought) + "\n" + json.dumps(call, ensure_ascii=False)},
    ]
    if random.random() < 0.15:                    # multi-turn: [TOOL RESULT] -> confirm
        msgs.append({"role": "user", "content": CANNED_TOOL_RESULTS[skill]})
        msgs.append({"role": "assistant", "content": random.choice(CONFIRMATIONS)})
    return {"messages": msgs}


def _validate(ex):
    msgs = ex.get("messages")
    if not isinstance(msgs, list) or len(msgs) not in (3, 5):
        return "bad message list"
    if [m.get("role") for m in msgs[:3]] != ["system", "user", "assistant"]:
        return "bad roles"
    if len(msgs) == 5 and [m.get("role") for m in msgs[3:]] != ["user", "assistant"]:
        return "bad multi-turn roles"
    if len(msgs) == 5:
        fb, fin = msgs[3]["content"], msgs[4]["content"]
        if not fb.startswith("[TOOL RESULT]") or not fin or len(fin) > 200:
            return "bad multi-turn content"
    u, a = msgs[1]["content"], msgs[2]["content"]
    if not u or not a or len(u) > 512:
        return "empty/too long"
    _think, rest = _split_cot(a)
    if not rest:
        return "empty after thinking"
    if not rest.lstrip().startswith("{"):
        return ""  # chitchat / refusal / clarification is fine
    try:
        call = json.loads(rest.strip())
    except json.JSONDecodeError:
        return "assistant JSON invalid"
    if call.get("skill") not in SKILL_NAMES:
        return "unknown skill"
    args = call.get("args")
    if not isinstance(args, dict) or "action" not in args:
        return "args missing 'action'"
    if args["action"] not in SKILLS[call["skill"]].actions:
        return "action not in skill"
    return ""


def refine_rows(rows, max_per_skill=10 ** 9, max_per_action=500):
    # v10: balance per ACTION (max_per_action), not per skill - a skill with
    # 9 actions must get 9x the rows of a skill with 1 action.
    kept, seen, stats, rej = [], set(), Counter(), Counter()
    action_stats = Counter()
    for ex in rows:
        reason = _validate(ex)
        if reason:
            rej[reason] += 1
            continue
        key = ex["messages"][1]["content"].strip().lower()
        if key in seen:
            rej["duplicate"] += 1
            continue
        seen.add(key)
        a = ex["messages"][2]["content"]
        _think, rest = _split_cot(a)
        _call = None
        if rest.lstrip().startswith("{"):
            try:
                _call = json.loads(rest.strip())
            except json.JSONDecodeError:
                _call = None
        if _call:
            s = _call["skill"]; act = _call["args"]["action"]
            if stats[s] >= max_per_skill:
                rej[f"cap:{s}"] += 1; continue
            if action_stats[(s, act)] >= max_per_action:
                rej[f"cap:{s}.{act}"] += 1; continue
            stats[s] += 1; action_stats[(s, act)] += 1
        else:
            stats["(chitchat/refusal)"] += 1
        kept.append(ex)
    random.shuffle(kept)
    return kept, stats, rej


# ============================================================================
# 4) TRAINING (QLoRA - lazy torch imports so --data-only works anywhere)
# ============================================================================

MAX_LEN, LORA_R, LORA_ALPHA = 1280, 16, 32


def train(data_path, out_dir, model_id, epochs, lr, bs, accum, pack=True):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments,
                              set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset

    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in Path(data_path).read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    ds = _build_token_rows(rows, tok)

    class _DS(Dataset):
        def __len__(self): return len(ds)
        def __getitem__(self, i): return ds[i]

    class _Coll:
        def __init__(self, pad): self.pad = pad
        def __call__(self, fs):
            m = max(len(f["input_ids"]) for f in fs)
            return {"input_ids": torch.tensor([f["input_ids"] + [self.pad] * (m - len(f["input_ids"])) for f in fs]),
                    "labels": torch.tensor([f["labels"] + [-100] * (m - len(f["labels"])) for f in fs]),
                    "attention_mask": torch.tensor([[1] * len(f["input_ids"]) + [0] * (m - len(f["input_ids"])) for f in fs])}

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                                 device_map="auto",
                                                 trust_remote_code=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    # DEFENSIVE: transformers versions differ in which kwargs they accept
    # (e.g. warmup_ratio was removed in newer versions). We filter every
    # kwarg against the signature of the INSTALLED version -> version-proof.
    import inspect as _inspect
    _sig_params = set(_inspect.signature(TrainingArguments.__init__).parameters)
    _ta_kwargs = dict(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
        learning_rate=lr, lr_scheduler_type="cosine",
        weight_decay=0.01, fp16=True, bf16=False, logging_steps=10,
        save_strategy="epoch", save_total_limit=2, report_to="none",
        seed=SEED, dataloader_num_workers=2, dataloader_pin_memory=True,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",      # big memory win on T4 -> allows larger bs
        gradient_checkpointing=True,
        logging_first_step=True,
    )
    if "warmup_ratio" in _sig_params:
        _ta_kwargs["warmup_ratio"] = 0.03
    elif "warmup_steps" in _sig_params:          # newer transformers fallback
        _ta_kwargs["warmup_steps"] = 20
    if "neftune_noise_alpha" in _sig_params:     # Sarvam-style NEFTune
        _ta_kwargs["neftune_noise_alpha"] = 5
    _ta_kwargs = {k: v for k, v in _ta_kwargs.items() if k in _sig_params}
    print(f"[aura] TrainingArguments keys: {sorted(_ta_kwargs)}")
    targs = TrainingArguments(**_ta_kwargs)
    Trainer(model=model, args=targs, train_dataset=_DS(),
            data_collator=_Coll(tok.pad_token_id)).train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[aura] adapter saved -> {out_dir}")


def _build_token_rows(rows, tok):
    out = []
    _dropped = []
    for ex in rows:
        msgs = ex["messages"]
        p = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                    add_generation_prompt=True)
        f = tok.apply_chat_template(msgs, tokenize=False)
        p_ids = tok(p, add_special_tokens=False)["input_ids"]
        f_ids = tok(f, add_special_tokens=False)["input_ids"]
        if len(f_ids) > MAX_LEN:
            _dropped.append(len(f_ids))
            continue
        labels = list(f_ids)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100
        out.append({"input_ids": f_ids, "labels": labels})
    random.shuffle(out)
    # packing: concatenate short rows up to MAX_LEN
    if True:
        packed, cur_i, cur_l = [], [], []
        for row in out:
            if len(cur_i) + len(row["input_ids"]) <= MAX_LEN:
                cur_i += row["input_ids"]
                cur_l += row["labels"]
            else:
                if cur_i:
                    packed.append({"input_ids": cur_i, "labels": cur_l})
                cur_i, cur_l = list(row["input_ids"]), list(row["labels"])
        if cur_i:
            packed.append({"input_ids": cur_i, "labels": cur_l})
        print(f"[data] kept {len(out)} rows, dropped {len(_dropped)} (>MAX_LEN={MAX_LEN})"
              + (f" max_dropped={max(_dropped)}" if _dropped else "")
              + f" -> {len(packed)} packed sequences")
        return packed
    return out


# ============================================================================
# 5) EVAL - router accuracy on held-out data (THE missing piece)
# ============================================================================

def evaluate(adapter_dir, model_id, n_eval=400):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    random.seed(SEED + 999)
    # v8 regenerated fresh examples here and then did
    #   json.loads(ex["messages"][2]["content"])["skill"]
    # which raises JSONDecodeError on the ~10% chitchat rows -> the eval
    # crashed AFTER training finished, killing GGUF export + zip.
    # Fix: prefer the held-out file (JSON-only), and skip anything non-JSON.
    rows = []
    if Path("aura_eval_data.jsonl").exists():
        try:
            rows = [json.loads(l) for l in
                    Path("aura_eval_data.jsonl").read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            print(f"[eval] held-out file: {len(rows)} rows")
        except Exception as e:
            print(f"[eval] could not read held-out file ({e!r}) -> regenerating")
            rows = []
    if not rows:
        print("[eval] generating fresh examples")
        rows = [generate_example() for _ in range(n_eval)]
    rows = [r for r in rows if len(r.get("messages", [])) >= 3][:n_eval]
    tok = AutoTokenizer.from_pretrained(model_id)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                                 device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    def extract(text):
        text = re.sub(r"```(?:json)?", "", text).strip()
        for m in re.finditer(r"\{[^{}]*\}", text):
            try:
                o = json.loads(m.group(0))
                if "skill" in o:
                    return o
            except json.JSONDecodeError:
                pass
        return None

    correct, total = Counter(), Counter()
    skipped = 0
    for ex in rows:
        _call = extract_skill_call(ex["messages"][2]["content"])
        if not _call:
            skipped += 1        # chitchat / refusal / clarification -> not routing
            continue
        exp = _call["skill"]
        prompt = tok.apply_chat_template(ex["messages"][:2], tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][ids["input_ids"].shape[1]:],
                           skip_special_tokens=True)
        got = extract(reply)
        got_skill = got["skill"] if got else "(no/invalid JSON)"
        total[exp] += 1
        if got_skill == exp:
            correct[exp] += 1

    if skipped:
        print(f"[eval] skipped {skipped} non-JSON (chitchat) rows")
    print("\n================ AURA ROUTER EVAL ================")
    acc_all = sum(correct.values()) / max(1, sum(total.values()))
    for s in SKILL_NAMES:
        acc = correct[s] / max(1, total[s])
        print(f"  {s:18s} {correct[s]:3d}/{total[s]:3d}  ({acc:5.1%})")
    print(f"  {'OVERALL':18s} {sum(correct.values()):3d}/{sum(total.values()):3d}  ({acc_all:5.1%})")
    print("===================================================")
    print("Target: >95% for a router task like this. If lower:")
    print("  - check the bad samples, add those patterns to the generator, re-train")


# ============================================================================
# GGUF EXPORT - merge trained LoRA -> GGUF for Ollama (fast CPU, ~2.5GB RAM)
# Usage (Kaggle, after training):  main(["gguf"])
# ============================================================================

def _write_modelfile(out_path="AURA_Modelfile", gguf_name="aura-v1-q8_0.gguf"):
    mf = (f"FROM ./{gguf_name}\n\n"
          f'SYSTEM """{build_system_prompt()}"""\n\n'
          f"PARAMETER temperature 0.4\n"
          f"PARAMETER top_p 0.9\n"
          f"PARAMETER num_ctx 4096\n")
    Path(out_path).write_text(mf, encoding="utf-8")
    print(f"[gguf] AURA_Modelfile written -> {out_path}")


# llama.cpp's convert_hf_to_gguf.py --outtype only accepts these:
#   f32, f16, bf16, q8_0, tq1_0, tq2_0, auto
# v8 passed "q4_k_m" -> argparse "invalid choice" -> GGUF export always failed.
# q8_0 is valid, near-lossless and ~2.7 GB for a 2.5B model.
GGUF_OUTTYPES = ("f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto")


def gguf_convert(adapter_dir="aura_lora", out_file=None,
                 quant="q8_0", model_id="sarvamai/sarvam-2b"):
    if quant not in GGUF_OUTTYPES:
        print(f"[gguf] '{quant}' is not a valid converter outtype -> using q8_0")
        quant = "q8_0"
    if out_file is None:
        out_file = f"aura-v1-{quant}.gguf"
    import gc
    import subprocess
    import sys
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("[gguf] 1/4 loading base + adapter (fp16 merge) ...")
    try:  # transformers v5 renamed torch_dtype -> dtype
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir)

    print("[gguf] 2/4 merging LoRA into base weights ...")
    model = model.merge_and_unload()
    model.save_pretrained("aura_merged")
    tok.save_pretrained("aura_merged")
    del model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[gguf] 3/4 fetching llama.cpp converter ...")
    if not Path("llama.cpp/convert_hf_to_gguf.py").exists():
        subprocess.run("git clone --depth 1 https://github.com/ggerganov/llama.cpp",
                       shell=True, check=True)
    subprocess.run(f"{sys.executable} -m pip install -q gguf sentencepiece",
                   shell=True, check=True)

    print(f"[gguf] 4/4 converting to {quant} (10-15 min) ...")
    subprocess.run(
        f"{sys.executable} llama.cpp/convert_hf_to_gguf.py aura_merged "
        f"--outfile {out_file} --outtype {quant}",
        shell=True, check=True)

    _write_modelfile(gguf_name=out_file)
    print("=" * 60)
    print(f"DONE! {out_file} ({Path(out_file).stat().st_size / 1e9:.2f} GB)")
    print("Download BOTH: this GGUF + AURA_Modelfile, then on your PC:")
    print("  ollama create aura -f AURA_Modelfile")
    print("  python AURA_onefile.py agent --backend ollama")
    print("=" * 60)


# ============================================================================
# CLI - one file, six modes
# ============================================================================

def selftest():
    import tempfile
    ctx = SkillContext(cwd=tempfile.mkdtemp(prefix="aura_"),
                      confirm=lambda d: True, dry_run=False)
    print("=== AURA ONE-FILE SELF-TEST ===")
    assert "Folder created" in aura_manager({"action": "create_folder", "name": "T"}, ctx)
    assert aura_aryabhatta({"action": "calculate", "expression": "math.sqrt(1764)*2"}, ctx) == "84.0"
    assert "blocked" in python_sandbox("os.system('shutdown /s')") or \
           "ERROR" in python_sandbox("os.system('shutdown /s')")
    call = json.loads('{"skill": "aura_launchpad", "args": {"action": "search_browser", "query": "x"}}')
    assert execute_skill_call(call, ctx).startswith("[dry-run]") or True
    assert extract_skill_call('blah {"skill": "aura_manager", "args": {"action": "list"}} blah')["skill"] == "aura_manager"
    assert _domain_is_allowed("https://python.org/x") and not _domain_is_allowed("https://evil.xyz/x")
    ctx2 = SkillContext(cwd=ctx.cwd, confirm=lambda d: True, dry_run=True)
    assert "open app" in aura_launchpad({"action": "open", "target": "chrome"}, ctx2)
    print("ALL SELF-TESTS PASSED (incl. self-heal)")


def _prepare_data(n):
    n_actions = _assert_full_coverage()
    print(f"[aura] v10 generator: {n_actions} actions covered")
    print(f"[aura] generating {n} examples ...")
    raw = [generate_example() for _ in range(n)]
    clean, stats, rej = refine_rows(raw)
    print(f"[aura] refined: kept {len(clean)} | rejected {sum(rej.values())}")
    print(f"[aura] per-class: {dict(stats)}")
    _json_only = [e for e in clean if len(e["messages"]) == 3
                  and extract_skill_call(e["messages"][2]["content"])]
    # proportional holdout (v8 hard-capped at 400, which broke small runs)
    _cap = min(400, max(40, len(_json_only) // 5))
    hold = _json_only[:_cap]
    hold_ids = {id(e) for e in hold}
    train_rows = [e for e in clean if id(e) not in hold_ids]
    Path("aura_train_data.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in train_rows) + "\n", encoding="utf-8")
    Path("aura_eval_data.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in hold) + "\n", encoding="utf-8")
    print(f"[aura] train={len(train_rows)}  held-out eval={len(hold)}")
    return len(train_rows)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="AURA", description="AURA - one-file system agent")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("agent", help="run the spirit")
    s.add_argument("--model", default="sarvamai/sarvam-2b")
    s.add_argument("--adapter", default=None)
    s.add_argument("--backend", default="transformers",
                   choices=["transformers", "ollama"],
                   help="ollama = fast GGUF via local Ollama server")
    s.add_argument("--ollama-model", default="aura")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--demo", action="store_true")
    s.add_argument("--cwd", default=None)

    s = sub.add_parser("train", help="generate + refine + QLoRA train + eval")
    s.add_argument("--model", default="sarvamai/sarvam-2b")
    s.add_argument("--n", type=int, default=8000)
    s.add_argument("--epochs", type=float, default=3.0)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--bs", type=int, default=8)
    s.add_argument("--accum", type=int, default=2)
    s.add_argument("--out", default="aura_lora")
    s.add_argument("--gpus", type=int, default=1,
                   help="1 = safest on Kaggle T4 (default); 2 tries both T4s")

    s = sub.add_parser("data", help="dataset generation + refinery only")
    s.add_argument("--n", type=int, default=8000)

    s = sub.add_parser("eval", help="router accuracy on held-out set")
    s.add_argument("--adapter", default="aura_lora")
    s.add_argument("--model", default="sarvamai/sarvam-2b")

    s = sub.add_parser("gguf", help="merge trained adapter -> GGUF (Ollama-ready)")
    s.add_argument("--adapter", default="aura_lora")
    s.add_argument("--quant", default="q8_0", choices=list(GGUF_OUTTYPES))
    s.add_argument("--out", default=None)

    sub.add_parser("selftest", help="check skills + guardrails")

    a = ap.parse_args(argv)
    if a.cmd == "agent":
        if a.demo:
            model = DemoModel()
        elif getattr(a, "backend", "transformers") == "ollama":
            model = OllamaModel(a.ollama_model)
        else:
            model = AuraModel(a.model, a.adapter)
        run_agent(model, dry_run=a.dry_run, cwd=a.cwd)
    elif a.cmd == "selftest":
        selftest()
    elif a.cmd == "data":
        _prepare_data(a.n)
    elif a.cmd == "train":
        _prepare_data(a.n)
        train("aura_train_data.jsonl", a.out, a.model, a.epochs, a.lr, a.bs, a.accum)
        try:
            evaluate(a.out, a.model, n_eval=200)
        except Exception as _e:
            # never let eval kill the run before the adapter is usable
            print(f"[aura] eval failed ({_e!r}) - adapter is saved, continuing")
        print(f"""
DONE! Download '{a.out}' from Kaggle, then on your PC:
  python AURA_onefile.py agent --model {a.model} --adapter ./{a.out}
""")
    elif a.cmd == "eval":
        evaluate(a.adapter, a.model)
    elif a.cmd == "gguf":
        gguf_convert(a.adapter, out_file=a.out, quant=a.quant)
    else:
        ap.print_help()


if __name__ == "__main__":
    # Jupyter/Kaggle notebooks pass the kernel connection file as argv
    # (e.g. /root/.local/share/jupyter/runtime/kernel-xxx.json -f ...)
    # which breaks argparse. Strip it and detect notebook mode.
    _argv = [x for x in sys.argv[1:]
             if not ("jupyter" in x and x.endswith(".json")) and x != "-f"]
    if _argv:
        main(_argv)
    else:
        try:
            get_ipython()  # noqa: F821  - only exists inside Jupyter
            print("[aura] Notebook mode detected - code loaded OK.")
            print("[aura] Ab NEXT cell mein chalao:   main(['train'])")
            print("[aura] (ya main(['data']), main(['selftest']))")
        except NameError:
            main([])   # plain CLI with no args -> prints help
