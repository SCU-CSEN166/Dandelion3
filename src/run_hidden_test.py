
from pathlib import Path
import json
import whisper

# Project paths
ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "hidden_test_audio"
OUTPUT_DIR = ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "hidden_test_transcripts.json"

def main():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio folder not found: {AUDIO_DIR}"
        )

    audio_files = sorted(AUDIO_DIR.glob("msg*.mp4"))

    if len(audio_files) != 10:
        raise ValueError(
            f"Expected 10 MP4 files, found {len(audio_files)}"
        )

    print("Loading Whisper model...")
    model = whisper.load_model("base")

    results = []

    for audio_file in audio_files:
        print(f"\nTranscribing {audio_file.name}...")

        try:
            transcription = model.transcribe(
                str(audio_file),
                language="en"
            )

            text = transcription["text"].strip()

            results.append({
                "message_id": audio_file.stem,
                "audio_file": audio_file.name,
                "transcript": text
            })

            print("Transcript:", text)

        except Exception as error:
            print(f"ERROR processing {audio_file.name}: {error}")

            results.append({
                "message_id": audio_file.stem,
                "audio_file": audio_file.name,
                "transcript": "",
                "error": str(error)
            })

    OUTPUT_DIR.mkdir(exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8"
    )

    print("\nTranscription finished.")
    print(f"Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()