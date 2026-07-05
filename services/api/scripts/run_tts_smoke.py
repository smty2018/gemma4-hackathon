import argparse
import asyncio
from pathlib import Path

from app.core.config import settings
from app.tts import SarvamStreamingTTS, SarvamTTSRequest

SAMPLES = {
    "en-IN": "Your document has been reviewed. Please check the required actions.",
    "hi-IN": "आपके दस्तावेज़ की जाँच हो गई है। कृपया आवश्यक कार्य देखें।",
    "bn-IN": "আপনার নথি পরীক্ষা করা হয়েছে। অনুগ্রহ করে প্রয়োজনীয় কাজগুলো দেখুন।",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paid Sarvam Bulbul v3 streaming smoke test in three languages."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker", default="shubh")
    return parser.parse_args()


async def run(output_dir: Path, speaker: str) -> None:
    if (
        settings.sarvam_api_key is None
        or not settings.sarvam_api_key.get_secret_value().strip()
    ):
        raise SystemExit("Set SARVAM_API_KEY in the repository .env file first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    service = SarvamStreamingTTS(settings.sarvam_api_key.get_secret_value())
    for language_code, text in SAMPLES.items():
        destination = output_dir / f"sarvam-{language_code}.mp3"
        with destination.open("wb") as audio_file:
            request = SarvamTTSRequest(
                text=text,
                target_language_code=language_code,
                speaker=speaker,
            )
            async for chunk in service.stream_audio(request):
                audio_file.write(chunk)
        print(f"{language_code}: {destination}")


def main() -> None:
    arguments = parse_arguments()
    asyncio.run(run(arguments.output_dir, arguments.speaker))


if __name__ == "__main__":
    main()
