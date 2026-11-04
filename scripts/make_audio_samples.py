"""Create sample voice notes with macOS text-to-speech so the voice path can be demoed and tested
without a microphone. (Real Telugu/Urdu recordings by volunteers should be added to data/audio_samples/.)

    python scripts/make_audio_samples.py
"""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "audio_samples"
SAMPLES = {
    # file name: (macOS voice, text)
    "hi_sewage_kukatpally.wav": ("Lekha", "कूकटपल्ली बस स्टॉप के पास नाला उफान पर है, पूरी सड़क पर गंदा पानी बह रहा है।"),
    "hi_live_wire_ameerpet.wav": ("Lekha", "अमीरपेट में बिजली का तार टूटकर सड़क पर गिरा हुआ है, बहुत खतरा है।"),
    "hi_no_water_dilsukhnagar.wav": ("Lekha", "दिलसुखनगर में चार दिन से पानी नहीं आया है। कृपया जल्दी कार्रवाई करें।"),
    "en_dog_bite_gachibowli.wav": ("Rishi", "A stray dog bit a small child near the bus stop in Gachibowli. "
                                            "Please send help."),
    "en_garbage_charminar.wav": ("Rishi", "Garbage has not been collected near Charminar for a week "
                                          "and the whole street stinks."),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (voice, text) in SAMPLES.items():
        aiff = OUT / (name + ".aiff")
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1",
                        str(OUT / name)], check=True)
        aiff.unlink()
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
