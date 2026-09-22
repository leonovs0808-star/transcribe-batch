"""
Пакетная расшифровка аудио/видео в текст. Движок по умолчанию — Deepgram nova-3:
он работает из любой страны, включая Россию, и сам размечает спикеров.

Для каждого файла:
  1. ffmpeg извлекает аудио в mp3 (mono 16kHz)
  2. аудио уходит в распознавание (для Groq — кусками, у него лимит ~25 МБ)
  3. результат сохраняется в .md рядом с исходником, с тем же именем

Записи длиннее трёх минут размечаются по спикерам ("Спикер 1: ..."), короткие —
сплошным текстом: короткая запись почти всегда мысль вслух, метки в ней мешают.
Флаги --speakers / --no-speakers перебивают это решение.

Безопасно перезапускать: если .md уже есть — файл пропускается.

Работает одинаково на macOS, Linux и Windows: нужен только Python 3.8+ и ffmpeg.
Никаких pip-пакетов и никакого curl — всё на стандартной библиотеке.

Запуск (на Windows пиши python вместо python3):
  Один файл:
    python3 transcribe.py "/путь/к/файлу.mp4"
  Вся папка (рекурсивно):
    python3 transcribe.py "/путь/к/папке"
  Только план (что будет обработано, без запуска):
    python3 transcribe.py "/путь/к/папке" --plan
  Принудительно с разделением по спикерам / принудительно без него:
    python3 transcribe.py "/путь/к/папке" --speakers
    python3 transcribe.py "/путь/к/папке" --no-speakers
  На движке Groq Whisper (бесплатен, но из России отдаёт HTTP 403):
    python3 transcribe.py "/путь/к/папке" --groq
  Проверить, что всё установлено:
    python3 transcribe.py --check
"""

from __future__ import annotations

import getpass
import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = platform.system() == "Windows"
PY = "python" if IS_WINDOWS else "python3"

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# Модели распознавания. Менять здесь — больше нигде не зашиты.
# Groq:     whisper-large-v3 — самая точная у Groq. Есть whisper-large-v3-turbo:
#           быстрее и дешевле, но заметно слабее на русском.
# Deepgram: nova-3 — новее и точнее nova-2 на русской живой речи (лучше пунктуация,
#           меньше выдуманных слов, чище разделение по спикерам).
GROQ_MODEL = "whisper-large-v3"
DEEPGRAM_MODEL = "nova-3"
MAX_BYTES = 24 * 1024 * 1024  # 24 МБ — лимит Groq
CHUNK_SECONDS = 600  # 10 минут на кусок (при 16kHz mono 32k mp3 ~ 2.4 МБ/10мин)
HTTP_TIMEOUT = 1800  # секунд на один запрос (длинный эфир уходит в Deepgram одним куском)

# Порог, после которого запись считается разговором, а не заметкой, и включается
# разделение по спикерам. Короткая запись — почти всегда диктофонная мысль вслух,
# метки «Спикер 1» в ней только мешают. Длинная — созвон, интервью, лекция.
# Перебивается флагами --speakers / --no-speakers.
SPEAKERS_THRESHOLD_SECONDS = 180

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
AUDIO_EXTS = (".ogg", ".mp3", ".m4a", ".aac", ".wav", ".opus", ".flac")

# Видеохостинги, с которых звук снимается через yt-dlp — напрямую файл оттуда не скачать.
YTDLP_HOSTS = ("youtube.com", "youtu.be", "vkvideo.ru", "vk.com/video", "vk.ru/video",
               "rutube.ru", "kinescope.io", "dzen.ru", "vimeo.com")


# ── Ключи из .env или окружения ───────────────────────────────────────────────
def read_env_value(name: str) -> str | None:
    """Ищет значение сначала в переменных окружения, потом в .env рядом со скриптом."""
    value = os.environ.get(name)
    if value:
        return value
    env_path = os.path.join(SCRIPT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def is_placeholder(value: str | None) -> bool:
    """Ключ из .env.example, который забыли заменить на настоящий."""
    return not value or value.startswith("gsk_твой") or value.startswith("твой")


def load_deepgram_key() -> str:
    key = read_env_value("DEEPGRAM_API_KEY")
    if not is_placeholder(key):
        return key
    print("ОШИБКА: DEEPGRAM_API_KEY не найден (ни в переменных окружения, ни в .env)")
    print("Получи бесплатный ключ на https://console.deepgram.com/ — карту не спрашивают,")
    print("на счёт сразу кладут $200, этого хватает примерно на 700 часов расшифровки.")
    print(f"Положи его в файл .env рядом со скриптом ({os.path.join(SCRIPT_DIR, '.env')}):")
    print("  DEEPGRAM_API_KEY=твой_ключ")
    sys.exit(1)


def load_groq_key() -> str:
    key = read_env_value("GROQ_API_KEY")
    if not is_placeholder(key):
        return key
    print("ОШИБКА: GROQ_API_KEY не найден (нужен для --groq).")
    print("Получи бесплатный ключ на https://console.groq.com/keys")
    print(f"и положи его в файл .env рядом со скриптом ({os.path.join(SCRIPT_DIR, '.env')}):")
    print("  GROQ_API_KEY=gsk_твой_ключ")
    print("Из России Groq не работает (HTTP 403) — там запускай без --groq, на Deepgram.")
    sys.exit(1)


# ── Проверка окружения ────────────────────────────────────────────────────────
def ffmpeg_install_hint() -> str:
    system = platform.system()
    if system == "Darwin":
        return "brew install ffmpeg"
    if system == "Windows":
        return ("winget install Gyan.FFmpeg   (или choco install ffmpeg, "
                "или вручную с https://www.gyan.dev/ffmpeg/builds/ — сборка essentials, "
                "папку bin добавить в PATH)")
    return "sudo apt install ffmpeg"


def require_tool(name: str):
    """Проверяет, что ffmpeg/ffprobe виден в PATH. Иначе — понятная ошибка вместо трейсбека."""
    if shutil.which(name) is None:
        print(f"ОШИБКА: {name} не найден в PATH.")
        print(f"Установи ffmpeg: {ffmpeg_install_hint()}")
        print("После установки открой терминал заново (PATH подхватывается при запуске).")
        sys.exit(1)


def proxy_note() -> str:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("ALL_PROXY")
    return f"да ({proxy})" if proxy else "нет (прямое подключение)"


def set_key(args: list[str]) -> int:
    """python3 transcribe.py --set-key deepgram [КЛЮЧ] — кладёт ключ в .env скрипта.

    Отдельная команда нужна, чтобы ключ не записывали самодельной строкой на питоне:
    такая строка считает путь от текущей папки, и при забытом cd ключ уходит в чужой
    .env — молча, с бодрым «ОК» в ответ. Здесь путь берётся от самого файла скрипта,
    промахнуться некуда. Без второго аргумента ключ спрашивается вводом и не проходит
    через чат с ИИ вообще.
    """
    known = {"deepgram": "DEEPGRAM_API_KEY", "groq": "GROQ_API_KEY"}
    i = args.index("--set-key")
    service = args[i + 1].lower() if len(args) > i + 1 else ""
    if service not in known:
        print("Использование: --set-key deepgram [КЛЮЧ]   (или groq вместо deepgram)")
        print("Без КЛЮЧА скрипт спросит его вводом — тогда ключ не попадёт в переписку.")
        return 1
    var = known[service]

    key = args[i + 2].strip() if len(args) > i + 2 else ""
    if not key:
        # getpass не печатает набранное на экран: ключ не останется ни в окне
        # терминала, ни в его логе, ни на скриншоте.
        try:
            key = getpass.getpass(f"Вставь {var} (ввод не отображается) и нажми Enter: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nОШИБКА: ключ не введён.")
            return 1
        except Exception:
            try:
                key = input(f"Вставь {var} и нажми Enter: ").strip()
            except EOFError:
                print("ОШИБКА: ключ не введён.")
                return 1
    if not key or is_placeholder(key):
        print("ОШИБКА: это не похоже на ключ.")
        return 1

    env_path = os.path.join(SCRIPT_DIR, ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    replaced = False
    for n, line in enumerate(lines):
        if line.strip().startswith(f"{var}="):
            lines[n] = f"{var}={key}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{var}={key}")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Ключ записан: {env_path}")
    print("Проверяю, принимает ли его сервис...", flush=True)
    ok, note = probe_key(service)
    print(f"{var}: {note}")
    return 0 if ok else 1


def probe_key(service: str) -> tuple[bool, str]:
    """Спрашивает у сервиса, принимает ли он ключ. Возвращает (годен, что сказать).

    Без этого проверка врала: она видела только, что в .env что-то вписано, и
    печатала «Готово к работе» на ключе с опечаткой. Человек шёл дальше и упирался
    в HTTP 401 уже на своём файле, не понимая, при чём тут ключ.
    """
    key = read_env_value(f"{service.upper()}_API_KEY")
    if is_placeholder(key):
        return False, "НЕ ЗАДАН"
    if service == "deepgram":
        url, headers = "https://api.deepgram.com/v1/projects", {"Authorization": f"Token {key}"}
    else:
        url, headers = "https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30):
            return True, "есть, ключ принят"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "вписан, но сервис его НЕ ПРИНЯЛ (401) — проверь, нет ли лишних пробелов, или создай новый"
        if e.code == 403:
            return False, "вписан, но сервис не обслуживает твою страну (403)"
        return False, f"вписан, но сервис ответил HTTP {e.code}"
    except Exception as e:
        # Нет сети — это не повод объявлять ключ негодным.
        return True, f"вписан, проверить не смог ({type(e).__name__}) — похоже, нет интернета"


def run_check():
    """python3 transcribe.py --check — показывает, что готово, а что нет."""
    print(f"ОС:        {platform.system()} {platform.release()}")
    print(f"Python:    {platform.python_version()} ({sys.executable})")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        print(f"{tool + ':':10} {path if path else 'НЕ НАЙДЕН — ' + ffmpeg_install_hint()}")
    print(f"Файл .env: {os.path.join(SCRIPT_DIR, '.env')}")
    print("Проверяю ключи у сервисов...", flush=True)

    deepgram_ok, deepgram_note = probe_key("deepgram")
    print(f"DEEPGRAM_API_KEY: {deepgram_note}"
          + ("" if deepgram_ok else " — ключ берётся на https://console.deepgram.com/"))
    groq_key = read_env_value("GROQ_API_KEY")
    if is_placeholder(groq_key):
        groq_ok = False
        print("GROQ_API_KEY:     не задан (нужен только для --groq, из России не работает)")
    else:
        groq_ok, groq_note = probe_key("groq")
        print(f"GROQ_API_KEY:     {groq_note}")
    print(f"Прокси:    {proxy_note()}")

    tools_ok = shutil.which("ffmpeg") and shutil.which("ffprobe")
    if tools_ok and (deepgram_ok or groq_ok):
        print("\nГотово к работе.")
        return 0
    print("\nЕсть чего не хватает — см. выше.")
    return 1


# ── HTTP на стандартной библиотеке (без curl) ─────────────────────────────────
def http_post(url: str, headers: dict, body: bytes, retries: int = 3) -> bytes:
    """POST с ретраями. Прокси берётся из HTTPS_PROXY/ALL_PROXY автоматически."""
    last_error = ""
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            last_error = f"HTTP {e.code}: {detail}"
            if e.code in (400, 401, 403, 404, 413):
                break  # ретраи не помогут — ключ, лимит или формат
        except Exception as e:  # сеть, таймаут, прокси
            last_error = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last_error or "запрос не удался")


def build_multipart(fields: dict, file_path: str, file_field: str = "file") -> tuple[bytes, str]:
    """Собирает multipart/form-data тело вручную — чтобы не тянуть requests."""
    boundary = "----transcribebatch" + uuid.uuid4().hex
    crlf = b"\r\n"
    parts = []
    for name, value in fields.items():
        parts.append(b"--" + boundary.encode() + crlf)
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode() + crlf + crlf)
        parts.append(str(value).encode("utf-8") + crlf)

    filename = os.path.basename(file_path)
    ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(file_path, "rb") as f:
        payload = f.read()
    parts.append(b"--" + boundary.encode() + crlf)
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'.encode()
        + crlf
    )
    parts.append(f"Content-Type: {ctype}".encode() + crlf + crlf)
    parts.append(payload + crlf)
    parts.append(b"--" + boundary.encode() + b"--" + crlf)
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# ── Текст ─────────────────────────────────────────────────────────────────────
def add_paragraphs(text: str, sentences_per_paragraph: int = 4) -> str:
    """Режет сплошной текст на абзацы: каждые N предложений = новый абзац.
       Последовательности типа '?!' или '...' считаются одним терминатором."""
    text = text.strip()
    if not text:
        return text

    sentences = []
    current = []
    chars = list(text)
    terminators = {".", "!", "?", "…"}
    for i, ch in enumerate(chars):
        current.append(ch)
        if ch in terminators:
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt not in terminators:
                s = "".join(current).strip()
                if s:
                    sentences.append(s)
                current = []
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)

    if len(sentences) <= 1:
        return text

    paragraphs = [
        " ".join(sentences[i:i + sentences_per_paragraph])
        for i in range(0, len(sentences), sentences_per_paragraph)
    ]
    return "\n\n".join(paragraphs)


def looks_like_our_transcript(md_path: str, media_path: str) -> bool:
    """Свою расшифровку узнаём по первой строке — скрипт пишет её как '# <имя файла>'."""
    title = os.path.splitext(os.path.basename(media_path))[0]
    try:
        with open(md_path, encoding="utf-8") as f:
            return f.readline().strip() == f"# {title}"
    except OSError:
        return False


def output_path_for(media_path: str) -> tuple[str, bool]:
    """Возвращает (куда писать, сделано ли уже).

    Рядом с медиа может лежать чужой .md — конспект урока, заметка. Его не трогаем:
    пишем в <имя>.расшифровка.md. А вот свою собственную расшифровку узнаём по первой
    строке и второй раз не делаем. Без этой проверки повторный запуск на той же папке
    расшифровывал всё заново, в обход обещанного пропуска, и платил за это дважды.
    """
    base, _ = os.path.splitext(media_path)
    plain = base + ".md"
    suffixed = base + ".расшифровка.md"
    if not os.path.exists(plain):
        return plain, False
    if looks_like_our_transcript(plain, media_path):
        return plain, True
    return suffixed, os.path.exists(suffixed)


# ── ffmpeg ────────────────────────────────────────────────────────────────────
# ── Ссылки: YouTube и прочие видео, Яндекс.Диск, Google Drive, прямой файл ────
def looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def is_ytdlp_url(url: str) -> bool:
    """Видеохостинг, откуда файл забирается через yt-dlp, а не прямой загрузкой."""
    return any(h in url for h in YTDLP_HOSTS)


def http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "transcribe-batch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def safe_filename(name: str) -> str:
    """Убирает из имени всё, на чём спотыкается файловая система."""
    name = re.sub(r'[\\/:*?"<>|]+', " ", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name[:150] or "запись"


def name_from_headers(resp) -> str | None:
    """Вытаскивает имя файла из Content-Disposition — у Дисков оно только там."""
    cd = resp.headers.get("Content-Disposition") or ""
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    if not m:
        return None
    return safe_filename(urllib.parse.unquote(m.group(1).strip()))


def download_stream(url: str, out_path: str) -> str | None:
    """Качает файл по прямой ссылке, показывая прогресс — большие видео идут долго.
       Возвращает имя файла, если сервер его подсказал."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 transcribe-batch"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp, open(out_path, "wb") as f:
        suggested = name_from_headers(resp)
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        last_shown = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got - last_shown >= 5 * 1024 * 1024:  # отчитываемся раз в 5 МБ
                last_shown = got
                if total:
                    print(f"      скачано {got // 1048576} из {total // 1048576} МБ", flush=True)
                else:
                    print(f"      скачано {got // 1048576} МБ", flush=True)
    print(f"      скачано {os.path.getsize(out_path) // 1048576} МБ, готово", flush=True)
    return suggested


def resolve_gdrive(file_id: str) -> str:
    """Прямая ссылка на файл Google Drive.
       На больших файлах Google сначала отдаёт HTML со страницей подтверждения —
       тогда собираем адрес из формы на этой странице."""
    initial = (f"https://drive.usercontent.google.com/download"
               f"?id={file_id}&export=download&confirm=t")
    req = urllib.request.Request(initial, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if "text/html" not in ctype:
                return initial
            page = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError:
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1"

    m = re.search(r'action="(https://drive\.usercontent\.google\.com/download[^"]+)"', page)
    if m:
        action = m.group(1).replace("&amp;", "&")
        fields = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page))
        params = urllib.parse.urlencode(fields)
        sep = "&" if "?" in action else "?"
        return f"{action}{sep}{params}"
    return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1"


def download_with_ytdlp(url: str, out_dir: str) -> str:
    """Снимает звуковую дорожку с видеохостинга. Возвращает путь к файлу.

    Дорожка берётся как есть (bestaudio), без перегона в mp3: лишнее сжатие
    портит разделение по спикерам — та же причина, что в prepare_for_deepgram.
    """
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "для ссылок на видеохостинги нужен yt-dlp, его нет.\n"
            f"      Поставь:  {PY} -m pip install -U yt-dlp\n"
            "      Ответит «externally-managed-environment» (свежие Ubuntu/Debian, питон\n"
            f"      из Homebrew) — добавь флаг:  {PY} -m pip install -U --break-system-packages yt-dlp\n"
            "      Через pipx ставить НЕ надо: скрипт подключает yt-dlp как библиотеку,\n"
            "      а pipx прячет её в отдельное окружение, и она не подхватится.\n"
            "      Ссылки на Яндекс.Диск, Google Drive и прямые файлы работают без него."
        )
    import yt_dlp

    template = os.path.join(out_dir, "%(title)s.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,  # иначе полоса загрузки затирает строки нашего вывода
        "socket_timeout": 60,
        "retries": 3,
        # YouTube регулярно меняет защиту — эти клиенты переживают её лучше прочих.
        "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
    }
    cookies = os.path.join(SCRIPT_DIR, "youtube_cookies.txt")
    if os.path.exists(cookies):
        opts["cookiefile"] = cookies

    before = set(os.listdir(out_dir))
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    created = [f for f in os.listdir(out_dir) if f not in before]
    if not created:
        raise RuntimeError("yt-dlp ничего не скачал")
    return os.path.join(out_dir, created[0])


def download_url(url: str, out_dir: str) -> str:
    """Любая поддерживаемая ссылка -> локальный файл. Возвращает путь к нему."""
    if is_ytdlp_url(url):
        print("      источник: видеохостинг, снимаю звуковую дорожку...", flush=True)
        return download_with_ytdlp(url, out_dir)

    gd = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if gd:
        print("      источник: Google Drive", flush=True)
        direct = resolve_gdrive(gd.group(1))
        out = os.path.join(out_dir, "gdrive-файл")
        suggested = download_stream(direct, out)
        if suggested:
            # Настоящее имя Google отдаёт только в заголовке ответа — берём его,
            # иначе все скачанные файлы звались бы одинаково.
            renamed = os.path.join(out_dir, suggested)
            os.rename(out, renamed)
            return renamed
        return out

    if any(h in url for h in ("disk.yandex.", "yadi.sk", "disk.360.yandex.")):
        print("      источник: Яндекс.Диск", flush=True)
        api = "https://cloud-api.yandex.net/v1/disk/public/resources"
        key = urllib.parse.quote(url, safe="")
        meta = http_get_json(f"{api}?public_key={key}")
        if meta.get("type") == "dir":
            raise RuntimeError(
                "это ссылка на ПАПКУ Яндекс.Диска, а не на файл. "
                "Дай ссылку на конкретный файл — либо скачай папку себе и укажи путь к ней."
            )
        name = safe_filename(meta.get("name") or "яндекс-диск-файл")
        href = http_get_json(f"{api}/download?public_key={key}")["href"]
        out = os.path.join(out_dir, name)
        download_stream(href, out)
        return out

    print("      источник: прямая ссылка на файл", flush=True)
    name = safe_filename(os.path.basename(urllib.parse.urlparse(url).path)) or "файл-по-ссылке"
    out = os.path.join(out_dir, name)
    download_stream(url, out)
    return out


def run_ffmpeg(cmd: list[str]):
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg извлечение аудио упало: {(r.stderr or '')[-400:]}")


def extract_audio(video_path: str, out_mp3: str):
    """Извлекает аудио в mp3 mono 16kHz 32kbps — экономно, под лимит Groq в 25 МБ."""
    run_ffmpeg([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame",
        "-ar", "16000", "-ac", "1", "-b:a", "32k",
        out_mp3,
    ])


def prepare_for_deepgram(media_path: str, tmp_dir: str) -> tuple[str, str]:
    """Готовит файл для Deepgram, возвращает (путь, content-type).

    Здесь принципиально НЕ жмём звук. Повторное сжатие в mp3 стирает тембровую
    разницу между голосами, и движок перестаёт различать спикеров: проверено —
    одна и та же запись даёт двух спикеров в оригинале и одного после прогона
    через mp3 32 kbps. Поэтому:
      - аудиофайл уходит как есть, вообще без ffmpeg;
      - из видео звук вынимается в FLAC (сжатие без потерь) — точность та же,
        что у WAV, а весит вдвое меньше.
    У Deepgram лимит на файл 2 ГБ, экономить на качестве незачем.
    """
    ext = os.path.splitext(media_path)[1].lower()
    if ext in AUDIO_EXTS:
        ctype = mimetypes.guess_type(media_path)[0] or "application/octet-stream"
        return media_path, ctype
    out_flac = os.path.join(tmp_dir, "audio.flac")
    run_ffmpeg([
        "ffmpeg", "-y", "-i", media_path,
        "-vn", "-acodec", "flac", "-ar", "16000", "-ac", "1",
        out_flac,
    ])
    return out_flac, "audio/flac"


def get_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, errors="replace",
        )
        return float(r.stdout.strip())
    except (ValueError, OSError):
        return 0.0


def split_audio(mp3_path: str, chunk_dir: str) -> list[str]:
    """Режет mp3 на куски по CHUNK_SECONDS секунд. Возвращает пути по порядку."""
    pattern = os.path.join(chunk_dir, "chunk_%04d.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", mp3_path,
        "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
        "-c", "copy", pattern,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg нарезка упала: {(r.stderr or '')[-400:]}")
    return sorted(
        os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)
        if f.startswith("chunk_") and f.endswith(".mp3")
    )


# ── Groq ──────────────────────────────────────────────────────────────────────
def transcribe_chunk(chunk_path: str, key: str) -> str:
    """Отправляет один кусок в Groq Whisper, возвращает текст."""
    body, content_type = build_multipart(
        {
            "model": GROQ_MODEL,
            "response_format": "text",
            "temperature": "0",
            "language": "ru",
        },
        chunk_path,
    )
    raw = http_post(
        GROQ_URL,
        {"Authorization": f"Bearer {key}", "Content-Type": content_type},
        body,
    )
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        raise RuntimeError("Groq вернул пустой ответ")
    return text


def transcribe_video(video_path: str, key: str) -> str:
    """Полный путь: видео/аудио -> mp3 -> куски -> текст."""
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = os.path.join(tmp, "audio.mp3")
        extract_audio(video_path, mp3)

        if os.path.getsize(mp3) <= MAX_BYTES:
            return transcribe_chunk(mp3, key)

        chunk_dir = os.path.join(tmp, "chunks")
        os.makedirs(chunk_dir)
        chunks = split_audio(mp3, chunk_dir)
        parts = []
        for i, c in enumerate(chunks, 1):
            print(f"      кусок {i}/{len(chunks)}...", flush=True)
            parts.append(transcribe_chunk(c, key))
        return " ".join(parts)


# ── Deepgram (разделение по спикерам) ─────────────────────────────────────────
def transcribe_deepgram(audio_path: str, key: str, diarize: bool, ctype: str) -> str:
    """Отправляет аудио в Deepgram nova-3, возвращает текст.
       diarize=True — реплики размечаются как "Спикер N:". Если движок нашёл в записи
       только один голос, меток не будет: размечать монолог нечем и незачем."""
    with open(audio_path, "rb") as f:
        payload = f.read()
    url = (f"{DEEPGRAM_URL}?model={DEEPGRAM_MODEL}&detect_language=true"
           f"&punctuate=true&diarize={'true' if diarize else 'false'}"
           f"&utterances={'true' if diarize else 'false'}")
    raw = http_post(
        url,
        {"Authorization": f"Token {key}", "Content-Type": ctype},
        payload,
    )
    data = json.loads(raw.decode("utf-8", "replace"))

    utterances = data.get("results", {}).get("utterances") or []
    if not utterances:
        return data["results"]["channels"][0]["alternatives"][0]["transcript"]

    if len({u["speaker"] for u in utterances}) == 1:
        return " ".join(u["transcript"] for u in utterances)

    blocks = []
    for u in utterances:
        if blocks and blocks[-1][0] == u["speaker"]:
            blocks[-1][1].append(u["transcript"])
        else:
            blocks.append((u["speaker"], [u["transcript"]]))

    return "\n\n".join(f"Спикер {speaker + 1}: {' '.join(parts)}" for speaker, parts in blocks)


def transcribe_video_deepgram(media_path: str, key: str, diarize: bool) -> str:
    """Видео/аудио -> Deepgram. Аудио уходит как есть, из видео звук вынимается в FLAC."""
    with tempfile.TemporaryDirectory() as tmp:
        audio, ctype = prepare_for_deepgram(media_path, tmp)
        return transcribe_deepgram(audio, key, diarize, ctype)


# ── Обход папки ───────────────────────────────────────────────────────────────
def collect_targets(root: str) -> list[str]:
    """Все аудио и видео в папке, рекурсивно. Ничего не отсеивает молча:
       что лежит в папке — то и будет расшифровано, список видно по --plan."""
    targets = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS + VIDEO_EXTS:
                targets.append(os.path.join(dirpath, f))
    return sorted(targets)


def print_usage():
    print("Использование:")
    print(f'  {PY} transcribe.py "путь/к/файлу.mp4"')
    print(f'  {PY} transcribe.py "путь/к/папке"              (рекурсивно, все аудио/видео)')
    print(f'  {PY} transcribe.py "https://youtu.be/..."      (ссылка: видео, Диски, прямой файл)')
    print(f'  {PY} transcribe.py "путь/к/папке" --plan       (только план, без запуска)')
    print(f"  {PY} transcribe.py --check                     (проверить установку)")
    print(f"  {PY} transcribe.py --set-key deepgram          (вписать ключ; спросит вводом)")
    print()
    print("Ссылки понимает: YouTube, ВК Видео, Рутуб, Кинескоп, Дзен, Vimeo,")
    print("Яндекс.Диск, Google Drive, прямая ссылка на файл.")
    print(f'  {PY} transcribe.py "<ссылка>" --out "папка"    (куда положить расшифровку)')
    print()
    print("Разделение по спикерам включается само на записях длиннее "
          f"{SPEAKERS_THRESHOLD_SECONDS // 60} минут. Перебить вручную:")
    print(f'  {PY} transcribe.py "путь" --speakers          (всегда с «Спикер 1/2»)')
    print(f'  {PY} transcribe.py "путь" --no-speakers       (всегда сплошным текстом)')
    print()
    print("Движок по умолчанию — Deepgram (работает везде, в том числе из России).")
    print(f'  {PY} transcribe.py "путь" --groq              (Groq Whisper; из РФ отдаёт 403)')


def main():
    args = sys.argv[1:]
    if not args:
        print_usage()
        sys.exit(1)
    if "--set-key" in args:
        sys.exit(set_key(args))
    if "--check" in args:
        sys.exit(run_check())

    root = args[0]
    plan_only = "--plan" in args
    # --groq оставлен для тех, кто за пределами России: Whisper там работает и бесплатен.
    # По умолчанию движок Deepgram — он единственный, кто отвечает из РФ.
    # "--deepgram" принимаем молча: так звали движок в старых инструкциях.
    groq_mode = "--groq" in args
    # Разделение по спикерам: по умолчанию решает длительность записи, флаги перебивают.
    force_speakers = "--speakers" in args
    force_no_speakers = "--no-speakers" in args
    if force_speakers and force_no_speakers:
        print("ОШИБКА: --speakers и --no-speakers вместе не имеют смысла, выбери одно.")
        sys.exit(1)

    # Куда класть результат для ссылок: рядом с файлом положить нельзя — файла
    # на диске нет. Кладём в текущую папку, либо туда, куда сказали флагом --out.
    out_dir = "."
    if "--out" in args:
        i = args.index("--out")
        if i + 1 >= len(args):
            print("ОШИБКА: после --out нужен путь к папке.")
            sys.exit(1)
        out_dir = args[i + 1]
        if not os.path.isdir(out_dir):
            print(f"ОШИБКА: папки нет: {out_dir}")
            sys.exit(1)

    is_url = looks_like_url(root)
    if is_url:
        targets = [root]
        base_dir = out_dir
    elif os.path.isfile(root):
        targets = [root]
        base_dir = os.path.dirname(root) or "."
    elif os.path.isdir(root):
        targets = collect_targets(root)
        base_dir = root
    else:
        print(f"Не найдено: {root}")
        print("Это должен быть путь к файлу, путь к папке или ссылка (http/https).")
        sys.exit(1)

    groq_key = None
    deepgram_key = None
    if not plan_only:
        require_tool("ffmpeg")
        require_tool("ffprobe")
        if groq_mode:
            groq_key = load_groq_key()
            # Запасной движок: Groq откажет по стране — уйдём на Deepgram, если ключ есть.
            deepgram_key = read_env_value("DEEPGRAM_API_KEY")
            if is_placeholder(deepgram_key):
                deepgram_key = None
        else:
            deepgram_key = load_deepgram_key()

    print(f"Найдено файлов для транскрипции: {len(targets)}")
    print(f"Движок: {'Groq ' + GROQ_MODEL if groq_mode else 'Deepgram ' + DEEPGRAM_MODEL}")
    if force_speakers:
        print("Спикеры: размечаю везде (--speakers)")
    elif force_no_speakers:
        print("Спикеры: не размечаю (--no-speakers)")
    elif not groq_mode:
        print(f"Спикеры: размечаю на записях длиннее "
              f"{SPEAKERS_THRESHOLD_SECONDS // 60} мин (короткие — сплошным текстом)")
    print(f"Прокси: {proxy_note()}\n")

    if plan_only:
        for i, path in enumerate(targets, 1):
            if is_url:
                print(f"[{i:2}] обработать   {path}")
                continue
            _, already = output_path_for(path)
            mark = "уже есть" if already else "обработать"
            rel = os.path.relpath(path, base_dir)
            print(f"[{i:2}] {mark:12} {rel}")
        print("\n(это только план, транскрипция не запущена — убери --plan чтобы запустить)")
        return

    done, skipped, failed = 0, 0, 0
    for i, source in enumerate(targets, 1):
        # Ссылку сначала превращаем в файл на диске, дальше путь общий для всех.
        downloaded_dir = None
        if is_url:
            print(f"[{i}/{len(targets)}] {source}", flush=True)
            downloaded_dir = tempfile.TemporaryDirectory()
            try:
                path = download_url(source, downloaded_dir.name)
            except Exception as e:
                print(f"      ОШИБКА: {e}")
                failed += 1
                downloaded_dir.cleanup()
                continue
            title = os.path.splitext(os.path.basename(path))[0]
            md_path = os.path.join(out_dir, title + ".md")
            already = os.path.exists(md_path)
            if already:
                print(f"      ПРОПУСК (расшифровка уже есть): {os.path.basename(md_path)}")
                skipped += 1
                downloaded_dir.cleanup()
                continue
        else:
            path = source
            md_path, already = output_path_for(path)
            if already:
                print(f"[{i}/{len(targets)}] ПРОПУСК (расшифровка уже есть): "
                      f"{os.path.basename(path)}")
                skipped += 1
                continue

        name = os.path.basename(path)
        dur = get_duration(path)
        # Разговор или заметка — решает длительность, пока флаг не сказал иначе.
        diarize = force_speakers or (
            not force_no_speakers and dur > SPEAKERS_THRESHOLD_SECONDS
        )
        if is_url:
            print(f"      {name}  (~{dur/60:.0f} мин)", flush=True)
        else:
            print(f"[{i}/{len(targets)}] {name}  (~{dur/60:.0f} мин)", flush=True)
        try:
            t0 = time.time()
            if not groq_mode:
                text = transcribe_video_deepgram(path, deepgram_key, diarize)
                if "Спикер " not in text:
                    text = add_paragraphs(text)
            else:
                try:
                    text = add_paragraphs(transcribe_video(path, groq_key))
                except RuntimeError as e:
                    if "HTTP 403" not in str(e) or not deepgram_key:
                        raise
                    # Groq не обслуживает страну пользователя — переходим на Deepgram.
                    # Настройки VPN/прокси пользователя при этом не трогаем.
                    print("      Groq недоступен из этой страны (403), перехожу на Deepgram...", flush=True)
                    groq_mode = False
                    text = transcribe_video_deepgram(path, deepgram_key, diarize)
                    if "Спикер " not in text:
                        text = add_paragraphs(text)
            title = os.path.splitext(name)[0]
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{text}\n")
            shown = os.path.abspath(md_path) if is_url else os.path.basename(md_path)
            print(f"      готово за {(time.time()-t0)/60:.1f} мин: {shown}")
            done += 1
        except Exception as e:
            text = str(e)
            if "HTTP 401" in text:
                print("      ОШИБКА: сервис не принял ключ (401). Ключ скопирован с опечаткой "
                      "или обрезан.")
                print(f"      Создай новый на https://console.deepgram.com/ и впиши его:")
                print(f"        {PY} transcribe.py --set-key deepgram")
            elif "HTTP 403" in text:
                print("      ОШИБКА: сервис не обслуживает твою страну (403).")
                print("      Для Groq это нормально в РФ — запускай без --groq, на Deepgram.")
            elif "HTTP 429" in text:
                print("      ОШИБКА: слишком много запросов подряд (429). Подожди минуту "
                      "и запусти снова — уже готовые файлы пропустятся.")
            else:
                print(f"      ОШИБКА: {e}")
            failed += 1
        finally:
            # Скачанное по ссылке не копим на диске — расшифровка уже сохранена.
            if downloaded_dir:
                downloaded_dir.cleanup()

    print(f"\nИтог: готово {done}, пропущено {skipped}, ошибок {failed}")
    # Ненулевой код, если что-то упало: чтобы прогон в фоне или по расписанию
    # не выглядел успешным при полностью провалившейся пачке.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
