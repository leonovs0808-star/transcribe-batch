"""
Пакетная транскрипция аудио/видео в текст через Groq Whisper (и опционально Deepgram
с разделением по спикерам).

Для каждого файла:
  1. ffmpeg извлекает аудио в mp3 (mono 16kHz)
  2. аудио режется на куски (лимит Groq ~25 МБ) и транскрибируется
  3. результат сохраняется в .md рядом с исходником, с тем же именем

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
  С разделением по спикерам (Deepgram nova-2 — для диалогов/интервью, нужен DEEPGRAM_API_KEY):
    python3 transcribe.py "/путь/к/папке" --speakers
  Проверить, что всё установлено:
    python3 transcribe.py --check
"""

from __future__ import annotations

import json
import mimetypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = platform.system() == "Windows"
PY = "python" if IS_WINDOWS else "python3"

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
MAX_BYTES = 24 * 1024 * 1024  # 24 МБ — лимит Groq
CHUNK_SECONDS = 600  # 10 минут на кусок (при 16kHz mono 32k mp3 ~ 2.4 МБ/10мин)
HTTP_TIMEOUT = 900  # секунд на один запрос

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
AUDIO_EXTS = (".ogg", ".mp3", ".m4a", ".aac", ".wav", ".opus")


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


def load_groq_key() -> str:
    key = read_env_value("GROQ_API_KEY")
    if key:
        return key
    print("ОШИБКА: GROQ_API_KEY не найден (ни в переменных окружения, ни в .env)")
    print("Получи бесплатный ключ на https://console.groq.com/keys")
    print(f"и положи его в файл .env рядом со скриптом ({os.path.join(SCRIPT_DIR, '.env')}):")
    print("  GROQ_API_KEY=gsk_твой_ключ")
    sys.exit(1)


def load_deepgram_key() -> str | None:
    return read_env_value("DEEPGRAM_API_KEY")


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


def run_check():
    """python3 transcribe.py --check — показывает, что готово, а что нет."""
    print(f"ОС:        {platform.system()} {platform.release()}")
    print(f"Python:    {platform.python_version()} ({sys.executable})")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        print(f"{tool + ':':10} {path if path else 'НЕ НАЙДЕН — ' + ffmpeg_install_hint()}")
    groq = read_env_value("GROQ_API_KEY")
    deepgram = read_env_value("DEEPGRAM_API_KEY")
    print(f"GROQ_API_KEY:     {'есть' if groq and not groq.startswith('gsk_твой') else 'НЕ ЗАДАН'}")
    print(f"DEEPGRAM_API_KEY: {'есть' if deepgram and not deepgram.startswith('твой') else 'не задан (нужен только для --speakers)'}")
    print(f"Прокси:    {proxy_note()}")
    ok = shutil.which("ffmpeg") and shutil.which("ffprobe") and groq and not groq.startswith("gsk_твой")
    print("\nГотово к работе." if ok else "\nЕсть чего не хватает — см. выше.")


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


def output_path_for(media_path: str) -> str:
    """Куда писать расшифровку. По умолчанию <имя>.md рядом с медиа.
       Если рядом уже лежит .md — добавляет .расшифровка.md, чтобы не перезаписать чужой файл."""
    base, _ = os.path.splitext(media_path)
    plain = base + ".md"
    if not os.path.exists(plain):
        return plain
    return base + ".расшифровка.md"


# ── ffmpeg ────────────────────────────────────────────────────────────────────
def extract_audio(video_path: str, out_mp3: str):
    """Извлекает аудио в mp3 mono 16kHz 32kbps."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame",
        "-ar", "16000", "-ac", "1", "-b:a", "32k",
        out_mp3,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg извлечение аудио упало: {(r.stderr or '')[-400:]}")


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
            "model": "whisper-large-v3",
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
def transcribe_deepgram(mp3_path: str, key: str) -> str:
    """Отправляет аудио в Deepgram nova-2 с diarization, возвращает текст.
       Монолог (один спикер занимает >=80% реплик) — без меток, диалог — с "Спикер N:"."""
    with open(mp3_path, "rb") as f:
        payload = f.read()
    url = (f"{DEEPGRAM_URL}?model=nova-2&detect_language=true"
           "&diarize=true&punctuate=true&utterances=true")
    raw = http_post(
        url,
        {"Authorization": f"Token {key}", "Content-Type": "audio/mpeg"},
        payload,
    )
    data = json.loads(raw.decode("utf-8", "replace"))

    utterances = data.get("results", {}).get("utterances") or []
    if not utterances:
        return data["results"]["channels"][0]["alternatives"][0]["transcript"]

    unique_speakers = {u["speaker"] for u in utterances}
    dominant_count = max(
        sum(1 for u in utterances if u["speaker"] == s) for s in unique_speakers
    )
    is_monologue = len(unique_speakers) == 1 or dominant_count / len(utterances) >= 0.8

    if is_monologue:
        return " ".join(u["transcript"] for u in utterances)

    blocks = []
    for u in utterances:
        if blocks and blocks[-1][0] == u["speaker"]:
            blocks[-1][1].append(u["transcript"])
        else:
            blocks.append((u["speaker"], [u["transcript"]]))

    return "\n\n".join(f"Спикер {speaker + 1}: {' '.join(parts)}" for speaker, parts in blocks)


def transcribe_video_speakers(video_path: str, key: str) -> str:
    """Видео/аудио -> mp3 -> Deepgram с разделением по спикерам."""
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = os.path.join(tmp, "audio.mp3")
        extract_audio(video_path, mp3)
        return transcribe_deepgram(mp3, key)


# ── Обход папки ───────────────────────────────────────────────────────────────
def collect_targets(root: str) -> list[str]:
    """Дедуп по папке:
       - если в папке есть аудио (.ogg и т.п.) — это главный урок; берём аудио,
         а самый большой mp4 (дубль главного урока) пропускаем,
         остальные mp4 (примеры, вебинары) берём;
       - если аудио в папке нет — берём все видео."""
    targets = []
    for dirpath, _, files in os.walk(root):
        audios = [os.path.join(dirpath, f) for f in files
                  if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
        videos = [os.path.join(dirpath, f) for f in files
                  if os.path.splitext(f)[1].lower() in VIDEO_EXTS]

        if audios:
            targets.extend(audios)
            if videos:
                main_video = max(videos, key=lambda p: os.path.getsize(p))
                targets.extend(v for v in videos if v != main_video)
        else:
            targets.extend(videos)
    return sorted(targets)


def print_usage():
    print("Использование:")
    print(f'  {PY} transcribe.py "путь/к/файлу.mp4"')
    print(f'  {PY} transcribe.py "путь/к/папке"          (рекурсивно, все аудио/видео)')
    print(f'  {PY} transcribe.py "путь/к/папке" --plan   (только план, без запуска)')
    print(f'  {PY} transcribe.py "путь/к/папке" --speakers  (с разделением по спикерам, Deepgram)')
    print(f"  {PY} transcribe.py --check                 (проверить установку)")


def main():
    args = sys.argv[1:]
    if not args:
        print_usage()
        sys.exit(1)
    if "--check" in args:
        run_check()
        return

    root = args[0]
    plan_only = "--plan" in args
    speakers_mode = "--speakers" in args

    if os.path.isfile(root):
        targets = [root]
        base_dir = os.path.dirname(root) or "."
    elif os.path.isdir(root):
        targets = collect_targets(root)
        base_dir = root
    else:
        print(f"Не найдено: {root}")
        sys.exit(1)

    groq_key = None
    deepgram_key = None
    if not plan_only:
        require_tool("ffmpeg")
        require_tool("ffprobe")
        if speakers_mode:
            deepgram_key = load_deepgram_key()
            if not deepgram_key:
                print("ОШИБКА: DEEPGRAM_API_KEY не найден (нужен для --speakers).")
                print("Получи ключ на https://console.deepgram.com/ и добавь в .env:")
                print("  DEEPGRAM_API_KEY=твой_ключ")
                sys.exit(1)
        else:
            groq_key = load_groq_key()

    print(f"Найдено файлов для транскрипции: {len(targets)}")
    print(f"Движок: {'Deepgram nova-2 (спикеры)' if speakers_mode else 'Groq Whisper (без спикеров)'}")
    print(f"Прокси: {proxy_note()}\n")

    if plan_only:
        for i, path in enumerate(targets, 1):
            md_path = output_path_for(path)
            mark = "уже есть" if os.path.exists(md_path) else "обработать"
            rel = os.path.relpath(path, base_dir)
            print(f"[{i:2}] {mark:12} {rel}")
        print("\n(это только план, транскрипция не запущена — убери --plan чтобы запустить)")
        return

    done, skipped, failed = 0, 0, 0
    for i, path in enumerate(targets, 1):
        md_path = output_path_for(path)
        name = os.path.basename(path)
        if os.path.exists(md_path):
            print(f"[{i}/{len(targets)}] ПРОПУСК (уже есть .md): {name}")
            skipped += 1
            continue

        dur = get_duration(path)
        print(f"[{i}/{len(targets)}] {name}  (~{dur/60:.0f} мин)", flush=True)
        try:
            t0 = time.time()
            if speakers_mode:
                text = transcribe_video_speakers(path, deepgram_key)
                if "Спикер " not in text:
                    text = add_paragraphs(text)
            else:
                text = add_paragraphs(transcribe_video(path, groq_key))
            title = os.path.splitext(name)[0]
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{text}\n")
            print(f"      готово за {(time.time()-t0)/60:.1f} мин: {os.path.basename(md_path)}")
            done += 1
        except Exception as e:
            print(f"      ОШИБКА: {e}")
            failed += 1

    print(f"\nИтог: готово {done}, пропущено {skipped}, ошибок {failed}")


if __name__ == "__main__":
    main()
