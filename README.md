# transcribe-batch

Скармливаешь скрипту видео/аудио или целую папку — рядом с каждым файлом появляется
`.md` с текстовой расшифровкой. Работает через Groq Whisper (`whisper-large-v3`) —
бесплатно, быстро, хорошо распознаёт русскую речь.

Один репозиторий на всё: **macOS, Linux и Windows**, **Claude Code и Codex**. Ставится
в папку навыков твоего ИИ — и дальше ты просто пишешь в чат «расшифруй вот эту папку»,
без команд.

## Не программист? Пусть поставит ИИ

Не открывай эту страницу руками — отправь **вот эту ссылку**:

> https://raw.githubusercontent.com/leonovs0808-star/transcribe-batch/master/INSTALL-FOR-AI.md

в чат с Claude, ChatGPT или Codex **на том же компьютере**, где будешь ставить скрипт
(нужен доступ к терминалу — на телефоне это не сработает), и напиши:
**«установи и настрой мне это по этой инструкции»**. Дальше ИИ проведёт тебя за руку:
определит твою систему, поставит недостающее, поможет получить бесплатный ключ и
запустит первую расшифровку.

Если хочешь сделать всё сам — читай дальше.

## Установка как навык (тогда работает по фразе в чате)

Репозиторий одновременно и обычный скрипт, и готовый навык. Клонируй его в папку
навыков своего агента — он подхватит `SKILL.md` сам:

**Claude Code** (macOS / Linux):
```bash
git clone https://github.com/leonovs0808-star/transcribe-batch.git ~/.claude/skills/transcribe-batch
```

**Codex** (macOS / Linux):
```bash
git clone https://github.com/leonovs0808-star/transcribe-batch.git ~/.codex/skills/transcribe-batch
```

**Windows (PowerShell)** — то же самое, только путь пишется так:
```powershell
git clone https://github.com/leonovs0808-star/transcribe-batch.git $env:USERPROFILE\.claude\skills\transcribe-batch
# или для Codex:
git clone https://github.com/leonovs0808-star/transcribe-batch.git $env:USERPROFILE\.codex\skills\transcribe-batch
```

Дальше зайди в эту папку, создай `.env` и впиши ключ (см. «Ключ» ниже). После этого в
любом чате достаточно написать «расшифруй /путь/к/папке» — агент сам вызовет скрипт.

## Что делает

- Проходит по папке (рекурсивно) или берёт один указанный файл
- Понимает `.mp4 .mov .mkv .webm .avi .m4v .mp3 .m4a .aac .wav .ogg .opus`
- Режет длинное аудио на куски и склеивает результат — лимиты Whisper не мешают
- Кладёт результат в `<имя файла>.md` рядом с исходником
- Уже расшифрованные файлы пропускает — safe перезапускать на той же папке
- Опционально: разделение по спикерам (`--speakers`, движок Deepgram) — для интервью,
  созвонов, диалогов, где важно кто что сказал

## Что нужно

- **Python 3.8+** — на Mac и Linux обычно уже стоит; на Windows — с
  https://www.python.org/downloads/ (при установке отметить галочку
  «Add python.exe to PATH»)
- **ffmpeg**:
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Windows: `winget install Gyan.FFmpeg` (или `choco install ffmpeg`, или вручную
    с https://www.gyan.dev/ffmpeg/builds/ — сборка essentials, папку `bin` добавить в PATH)
- Бесплатный ключ Groq — https://console.groq.com/keys
- (опционально) ключ Deepgram для `--speakers` — https://console.deepgram.com/

Никаких pip-пакетов ставить не надо: скрипт использует только стандартную библиотеку Python.

## Ключ

```bash
cp .env.example .env      # Windows PowerShell: copy .env.example .env
```

Открой `.env` и впиши свой Groq-ключ вместо `gsk_твой_ключ_отсюда_console.groq.com`.

## Проверка установки

```bash
python3 transcribe.py --check
```

Печатает ОС, версию Python, найден ли ffmpeg/ffprobe, задан ли ключ, есть ли прокси.
Чего не хватает — подсказка идёт в том же выводе.

## Запуск руками

На Windows везде пиши `python` вместо `python3`.

Один файл:
```bash
python3 transcribe.py "/путь/к/видео.mp4"
```

Вся папка (рекурсивно):
```bash
python3 transcribe.py "/путь/к/папке"
```

Только план, без запуска:
```bash
python3 transcribe.py "/путь/к/папке" --plan
```

С разделением по спикерам (нужен `DEEPGRAM_API_KEY` в `.env`):
```bash
python3 transcribe.py "/путь/к/папке" --speakers
```

## Если Groq не работает в твоей стране (в РФ — не работает)

Признак — ошибка `HTTP 403`. **VPN включать и настраивать не надо.** Возьми бесплатный
ключ Deepgram на https://console.deepgram.com/, впиши его в `.env` строкой
`DEEPGRAM_API_KEY=...` и запускай с флагом:

```bash
python3 transcribe.py "/путь/к/папке" --deepgram
```

Deepgram из России работает напрямую. Качество на русском сопоставимое.

Если в `.env` лежат оба ключа, переключение произойдёт само: скрипт получит 403 от Groq,
перейдёт на Deepgram и напишет об этом в выводе.

Остаться именно на Groq через свой прокси тоже можно:

```bash
HTTPS_PROXY=http://127.0.0.1:7897 python3 transcribe.py "/путь/к/папке"
```

В PowerShell — двумя командами: `$env:HTTPS_PROXY = "http://127.0.0.1:7897"`, затем запуск.
Порт зависит от твоего VPN, `7897` — просто пример.

## Частые вопросы

**Groq стоит денег?**
Бесплатный лимит покрывает обычное личное использование.

**Аудио уходит в облако — это безопасно?**
Groq (и Deepgram для `--speakers`) — облачные API, файл уходит на их серверы для
распознавания. Актуальную политику хранения данных смотри на их сайтах
(console.groq.com, deepgram.com) — для конфиденциальных материалов оценивай риски сам.

**Скрипт пишет, что ffmpeg не найден**
Проверь `ffmpeg -version` в терминале. Если только что установил — открой окно терминала
заново, PATH подхватывается при запуске.

**Долгий прогон на Windows**
`nohup` там нет — просто не закрывай окно, пока идёт расшифровка.

## Поддержка

Проблемы, вопросы — заводи issue в этом репозитории, либо пиши
[@SipitaSergey](https://t.me/SipitaSergey) в Telegram.
