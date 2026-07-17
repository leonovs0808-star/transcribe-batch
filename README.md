# transcribe-batch

Простой скрипт: скармливаешь ему видео/аудио или папку с ними — на выходе рядом
с каждым файлом появляется `.md` с текстовой расшифровкой. Работает через Groq
Whisper (`whisper-large-v3`) — бесплатно, быстро, хорошо распознаёт русскую речь.

Не программист и не знаешь, что со всем этим делать? Не открывай эту страницу
руками — просто отправь ссылку на неё в Claude или ChatGPT и напиши:
**"установи и настрой мне это по инструкции INSTALL-FOR-AI.md"**. Дальше ИИ
проведёт тебя за руку: проверит, что нужно, поставит недостающее, поможет
получить бесплатный ключ и запустит первую расшифровку.

Если хочешь сделать всё сам — читай дальше.

## Пользуешься Claude Code?

Этот репозиторий одновременно и обычный скрипт, и готовый Claude Code скил.
Склонируй его прямо в `~/.claude/skills/transcribe-batch/` — Claude подхватит
`SKILL.md` и дальше можно просто писать в чате "расшифруй /путь/к/папке", без
ручного набора команд каждый раз.

```bash
git clone https://github.com/leonovs0808-star/transcribe-batch.git ~/.claude/skills/transcribe-batch
cd ~/.claude/skills/transcribe-batch
cp .env.example .env
```

Дальше впиши ключ в `.env` (см. раздел "Установка" ниже) — и скил готов.

## Что делает

- Проходит по папке (рекурсивно) или берёт один указанный файл
- Понимает `.mp4 .mov .mkv .webm .avi .m4v .mp3 .m4a .aac .wav .ogg .opus`
- Режет длинное аудио на куски и склеивает результат — лимиты Whisper не мешают
- Кладёт результат в `<имя файла>.md` рядом с исходником
- Уже расшифрованные файлы пропускает — safe перезапускать на той же папке
- Опционально: разделение по спикерам (`--speakers`, движок Deepgram) — для
  интервью, созвонов, диалогов, где важно кто что сказал

## Что нужно

- Python 3 (на Mac и Linux обычно уже стоит)
- ffmpeg (`brew install ffmpeg` на Mac, `apt install ffmpeg` на Linux, на
  Windows — через [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) или `choco install ffmpeg`)
- Бесплатный ключ Groq — [console.groq.com/keys](https://console.groq.com/keys)
- (опционально) ключ Deepgram для `--speakers` — [console.deepgram.com](https://console.deepgram.com/)

Больше ничего ставить не нужно — скрипт использует только стандартную
библиотеку Python и системный `curl`.

## Установка

```bash
git clone https://github.com/leonovs0808-star/transcribe-batch.git
cd transcribe-batch
cp .env.example .env
```

Открой `.env` и впиши свой Groq-ключ вместо `gsk_твой_ключ_отсюда_console.groq.com`.

## Запуск

Один файл:

```bash
python3 transcribe.py "/путь/к/видео.mp4"
```

Вся папка (рекурсивно, все аудио/видео внутри):

```bash
python3 transcribe.py "/путь/к/папке"
```

Только посмотреть план, без запуска:

```bash
python3 transcribe.py "/путь/к/папке" --plan
```

С разделением по спикерам (нужен `DEEPGRAM_API_KEY` в `.env`):

```bash
python3 transcribe.py "/путь/к/папке" --speakers
```

## Если Groq заблокирован в твоей сети

Запусти с прокси:

```bash
HTTPS_PROXY=http://127.0.0.1:7897 python3 transcribe.py "/путь/к/папке"
```

## Частые вопросы

**Groq стоит денег?**
Бесплатный лимит покрывает обычное личное использование.

**Аудио уходит в облако — это безопасно?**
Groq (и Deepgram для `--speakers`) — облачные API, файл уходит на их серверы
для распознавания. Актуальную политику хранения данных смотри на их сайтах
(console.groq.com, deepgram.com) — для конфиденциальных материалов оценивай
риски сам.

**Скрипт упал с ошибкой ffmpeg**
Проверь что ffmpeg установлен: `ffmpeg -version` в терминале. Если команда не
найдена — ffmpeg не установлен, см. раздел "Что нужно" выше.

## Поддержка

Проблемы, вопросы — заводи issue в этом репозитории, либо пиши
[@SipitaSergey](https://t.me/SipitaSergey) в Telegram.
