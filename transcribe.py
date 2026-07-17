"""
Пакетная транскрипция аудио/видео в текст через Groq Whisper (и опционально Deepgram
с разделением по спикерам).

Для каждого файла:
  1. ffmpeg извлекает аудио в mp3 (mono 16kHz)
  2. аудио режется на куски (лимит Groq ~25 МБ) и транскрибируется
  3. результат сохраняется в .md рядом с исходником, с тем же именем

Безопасно перезапускать: если .md уже есть — файл пропускается.

Запуск:
  Один файл:
    python3 transcribe.py "/путь/к/файлу.mp4"
  Вся папка (рекурсивно):
    python3 transcribe.py "/путь/к/папке"
  Только план (что будет обработано, без запуска):
    python3 transcribe.py "/путь/к/папке" --plan
  С разделением по спикерам (Deepgram nova-2 — для диалогов/интервью, нужен DEEPGRAM_API_KEY):
    python3 transcribe.py "/путь/к/папке" --speakers
"""

import os
import sys
import json
import subprocess
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Groq API key ──────────────────────────────────────────────────────────────
def load_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    env_path = os.path.join(SCRIPT_DIR, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    print("ERROR: GROQ_API_KEY не найден (ни в переменных окружения, ни в .env)")
    print("Получи бесплатный ключ на https://console.groq.com/keys и положи его в .env рядом со скриптом:")
    print('  GROQ_API_KEY=gsk_твой_ключ')
    sys.exit(1)


GROQ_KEY = load_groq_key()
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MAX_BYTES = 24 * 1024 * 1024  # 24 МБ — лимит Groq
CHUNK_SECONDS = 600  # 10 минут на кусок (при 16kHz mono 32k mp3 ~ 2.4 МБ/10мин)


# ── Deepgram API key (для --speakers) ──────────────────────────────────────────
def load_deepgram_key() -> str | None:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if key:
        return key
    env_path = os.path.join(SCRIPT_DIR, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("DEEPGRAM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
AUDIO_EXTS = (".ogg", ".mp3", ".m4a", ".aac", ".wav", ".opus")


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


# Прокси из окружения (curl сам подхватит HTTPS_PROXY); используем --noproxy только если он не задан
USE_PROXY = bool(os.environ.get("HTTPS_PROXY") or os.environ.get("ALL_PROXY"))


def extract_audio(video_path: str, out_mp3: str):
    """Извлекает аудио в mp3 mono 16kHz 32kbps."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame",
        "-ar", "16000", "-ac", "1", "-b:a", "32k",
        out_mp3,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg извлечение аудио упало: {r.stderr[-400:]}")


def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def split_audio(mp3_path: str, chunk_dir: str) -> list[str]:
    """Режет mp3 на куски по CHUNK_SECONDS секунд. Возвращает пути по порядку."""
    pattern = os.path.join(chunk_dir, "chunk_%04d.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", mp3_path,
        "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
        "-c", "copy", pattern,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg нарезка упала: {r.stderr[-400:]}")
    chunks = sorted(
        os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)
        if f.startswith("chunk_") and f.endswith(".mp3")
    )
    return chunks


def transcribe_chunk(chunk_path: str) -> str:
    """Отправляет один кусок в Groq Whisper, возвращает текст. 3 попытки."""
    args = [
        "curl", "-s", "--fail-with-body", "-X", "POST", GROQ_URL,
        "-H", f"Authorization: Bearer {GROQ_KEY}",
        "-F", "model=whisper-large-v3",
        "-F", "response_format=text",
        "-F", "temperature=0",
        "-F", "language=ru",
        "-F", f"file=@{chunk_path};type=audio/mpeg",
    ]
    if not USE_PROXY:
        args[1:1] = ["--noproxy", "*"]

    for attempt in range(3):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        if attempt < 2:
            time.sleep(2)
    raise RuntimeError(f"Groq не ответил: {r.stdout[-200:]} {r.stderr[-200:]}")


def transcribe_video(video_path: str) -> str:
    """Полный путь: видео/аудио → mp3 → куски → текст."""
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = os.path.join(tmp, "audio.mp3")
        extract_audio(video_path, mp3)

        size = os.path.getsize(mp3)
        if size <= MAX_BYTES:
            return transcribe_chunk(mp3)

        chunk_dir = os.path.join(tmp, "chunks")
        os.makedirs(chunk_dir)
        chunks = split_audio(mp3, chunk_dir)
        parts = []
        for i, c in enumerate(chunks, 1):
            print(f"      кусок {i}/{len(chunks)}...", flush=True)
            parts.append(transcribe_chunk(c))
        return " ".join(parts)


def transcribe_deepgram(mp3_path: str, key: str) -> str:
    """Отправляет аудио в Deepgram nova-2 с diarization, возвращает текст.
       Монолог (один спикер занимает ≥80% реплик) — без меток, диалог — с "Спикер N:"."""
    args = [
        "curl", "-s", "--fail-with-body", "-X", "POST",
        f"{DEEPGRAM_URL}?model=nova-2&detect_language=true&diarize=true&punctuate=true&utterances=true",
        "-H", f"Authorization: Token {key}",
        "-H", "Content-Type: audio/mpeg",
        "--data-binary", f"@{mp3_path}",
    ]
    if not USE_PROXY:
        args[1:1] = ["--noproxy", "*"]

    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Deepgram не ответил: {r.stdout[-300:]} {r.stderr[-300:]}")

    data = json.loads(r.stdout)
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
    """Видео/аудио → mp3 → Deepgram с разделением по спикерам."""
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = os.path.join(tmp, "audio.mp3")
        extract_audio(video_path, mp3)
        return transcribe_deepgram(mp3, key)


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


def main():
    if len(sys.argv) < 2:
        print('Использование:')
        print('  python3 transcribe.py "/путь/к/файлу.mp4"')
        print('  python3 transcribe.py "/путь/к/папке"          (рекурсивно, все аудио/видео)')
        print('  python3 transcribe.py "/путь/к/папке" --plan   (только план, без запуска)')
        print('  python3 transcribe.py "/путь/к/папке" --speakers   (с разделением по спикерам, Deepgram)')
        sys.exit(1)

    root = sys.argv[1]
    if os.path.isfile(root):
        targets = [root]
        base_dir = os.path.dirname(root) or "."
    elif os.path.isdir(root):
        targets = collect_targets(root)
        base_dir = root
    else:
        print(f"Не найдено: {root}")
        sys.exit(1)

    plan_only = "--plan" in sys.argv
    speakers_mode = "--speakers" in sys.argv

    deepgram_key = None
    if speakers_mode and not plan_only:
        deepgram_key = load_deepgram_key()
        if not deepgram_key:
            print("ERROR: DEEPGRAM_API_KEY не найден (нужен для --speakers).")
            print("Получи ключ на https://console.deepgram.com/ и добавь в .env:")
            print("  DEEPGRAM_API_KEY=твой_ключ")
            sys.exit(1)

    print(f"Найдено файлов для транскрипции: {len(targets)}")
    print(f"Движок: {'Deepgram nova-2 (спикеры)' if speakers_mode else 'Groq Whisper (без спикеров)'}")
    print(f"Прокси: {'да (' + (os.environ.get('HTTPS_PROXY') or os.environ.get('ALL_PROXY')) + ')' if USE_PROXY else 'нет (прямое подключение)'}\n")

    if plan_only:
        for i, path in enumerate(targets, 1):
            md_path = output_path_for(path)
            mark = "уже есть" if os.path.exists(md_path) else "→ обработать"
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
                text = add_paragraphs(transcribe_video(path))
            title = os.path.splitext(name)[0]
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{text}\n")
            print(f"      готово за {(time.time()-t0)/60:.1f} мин → {os.path.basename(md_path)}")
            done += 1
        except Exception as e:
            print(f"      ОШИБКА: {e}")
            failed += 1

    print(f"\nИтог: готово {done}, пропущено {skipped}, ошибок {failed}")


if __name__ == "__main__":
    main()
