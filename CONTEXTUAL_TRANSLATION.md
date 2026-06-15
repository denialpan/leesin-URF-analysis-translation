# Contextual Gameplay Translation

`generate_contextual_translation.py` uses a staged pipeline:

1. Whisper `large-v3` provides or recovers Chinese speech.
2. `qwen2.5:14b` reconstructs Chinese and produces the main English translation
   from audio-derived text, nearby dialogue, and the terminology glossary.
3. `qwen2.5vl:7b` sees gameplay frames and HUD context only for ambiguous cues.

The visual stage verifies the audio-first translation. It is not allowed to
reinterpret every sentence from gameplay.

## Offline Context Bundle

This extracts the evidence but makes no network requests:

```powershell
python .\generate_contextual_translation.py "path\to\video.mp4" `
  --provider bundle
```

Test a limited cue range:

```powershell
python .\generate_contextual_translation.py "path\to\video.mp4" `
  --provider bundle --start-cue 35 --end-cue 36
```

The generated `context-jobs.json` is suitable for manual review or a separate
multimodal translation process.

## Vision API

Use an OpenAI-compatible vision chat endpoint:

```powershell
$env:OPENAI_API_KEY = "your-api-key"

python .\generate_contextual_translation.py "path\to\video.mp4" `
  --provider api `
  --api-url "https://your-host/v1/chat/completions" `
  --model "your-vision-model"
```

Outputs include:

- `*.corrected-chinese.srt`
- `*.contextual-english.srt`
- `*.context-results.json`
- `*.review.json`
- `*.context-jobs.json`
- Five gameplay frames per cue

Results below `--review-threshold` or explicitly marked uncertain by the model
are written to the review queue. Existing API results are reused so an
interrupted run can resume.

For local Ollama, install both staged models:

```powershell
ollama pull qwen2.5:14b
ollama pull qwen2.5vl:7b
```

Some `qwen3-vl` builds remain in thinking mode despite `think=false`, consume
the response budget, and return no final JSON.

## Single-Video Pipeline

Build only the offline context bundle:

```powershell
python .\run_single_video_pipeline.py "path\to\video.mp4" `
  --skip-hud --contextual-translation bundle
```

Generate contextual English through an API:

```powershell
python .\run_single_video_pipeline.py "path\to\video.mp4" `
  --skip-hud `
  --contextual-translation api `
  --context-api-url "https://your-host/v1/chat/completions" `
  --context-model "your-vision-model"
```

Omit `--skip-hud` when HUD analysis dependencies and rules files are present.

Reuse an existing Chinese transcript and run only contextual translation:

```powershell
python .\run_single_video_pipeline.py "path\to\video.mp4" `
  --skip-hud `
  --skip-transcription `
  --contextual-translation api `
  --context-api-url "http://localhost:11434/v1/chat/completions" `
  --context-text-model "qwen2.5:14b" `
  --context-vision-model "qwen2.5vl:7b"
```

Analyze an inclusive source-video frame range and print the likely Chinese and
English after processing:

```powershell
python .\run_single_video_pipeline.py "path\to\video.mp4" `
  --skip-hud `
  --skip-transcription `
  --contextual-translation api `
  --context-api-url "http://localhost:11434/v1/chat/completions" `
  --context-text-model "qwen2.5:14b" `
  --context-vision-model "qwen2.5vl:7b" `
  --context-start-frame 9236 `
  --context-end-frame 9807
```

The script reads the source video's actual frame rate and includes every
Chinese cue that overlaps the requested range. It prints cue timestamps,
corrected Chinese, English, confidence, and review status.

If the selected frame range has no Chinese cues, the script automatically
retries speech recognition using:

- Raw filtered audio
- An 8 dB vocal boost
- A 16 dB vocal boost
- Dynamic volume normalization

It compares confidence and agreement between the four results, rejects common
Whisper hallucinations, and sends the selected recovered Chinese to the
contextual model. Disable this behavior with:

```powershell
--context-no-audio-recovery
```

Local Ollama requests show an indeterminate per-cue progress bar with elapsed
time, retry attempt, frame count, and streamed response size. Models do not
report a reliable completion percentage for an individual generation.

Edit `league-terminology.json` to add champion names, ability phrases, items,
or conversational translations. Its guidance explicitly prevents glossary
terms from overriding unsupported audio.
