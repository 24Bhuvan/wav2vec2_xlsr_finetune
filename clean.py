from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

AUDIO_DIR = RAW_DATA_DIR / "audio"
TRANSCRIPT_DIR = RAW_DATA_DIR / "transcripts"
ANNOTATION_DIR = RAW_DATA_DIR / "annotations"

REPORT_DIR = PROJECT_ROOT / "outputs" / "cleaning_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "cleaned_files.txt"

# =====================================================
# STATS
# =====================================================

bad_ids = []

deleted_annotations = 0
deleted_transcripts = 0
deleted_audio = 0

# =====================================================
# PASS 1
# Remove bad annotation + transcript + audio
# =====================================================

for ann_file in ANNOTATION_DIR.glob("*_corrected.txt"):

    try:
        content = ann_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"Could not read {ann_file.name}: {e}")
        continue

    is_bad = False

    for line in content.splitlines():

        parts = line.split()

        if len(parts) >= 3 and parts[2] == "#":
            is_bad = True
            break

    if not is_bad:
        continue

    base_name = ann_file.stem.replace("_corrected", "")

    transcript_file = TRANSCRIPT_DIR / f"{base_name}.txt"
    audio_file = AUDIO_DIR / f"{base_name}.mp3"

    print(f"\nRemoving bad set: {base_name}")

    if ann_file.exists():
        ann_file.unlink()
        deleted_annotations += 1

    if transcript_file.exists():
        transcript_file.unlink()
        deleted_transcripts += 1

    if audio_file.exists():
        audio_file.unlink()
        deleted_audio += 1

    bad_ids.append(base_name)

# =====================================================
# PASS 2
# Remove orphan transcripts
# =====================================================

valid_ids = {
    f.stem.replace("_corrected", "")
    for f in ANNOTATION_DIR.glob("*_corrected.txt")
}

orphan_transcripts = []

for txt_file in TRANSCRIPT_DIR.glob("*.txt"):

    if txt_file.stem not in valid_ids:

        orphan_transcripts.append(txt_file.stem)

        print(f"Removing orphan transcript: {txt_file.name}")

        txt_file.unlink()
        deleted_transcripts += 1

# =====================================================
# PASS 3
# Remove orphan audio
# =====================================================

orphan_audio = []

for audio_file in AUDIO_DIR.glob("*.mp3"):

    if audio_file.stem not in valid_ids:

        orphan_audio.append(audio_file.stem)

        print(f"Removing orphan audio: {audio_file.name}")

        audio_file.unlink()
        deleted_audio += 1

# =====================================================
# REPORT
# =====================================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("CLEANING REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Bad annotation sets removed : {len(bad_ids)}\n")
    f.write(f"Annotations deleted         : {deleted_annotations}\n")
    f.write(f"Transcripts deleted         : {deleted_transcripts}\n")
    f.write(f"Audio deleted               : {deleted_audio}\n\n")

    f.write("BAD FILE IDS\n")
    f.write("-" * 60 + "\n")

    for item in bad_ids:
        f.write(f"{item}\n")

    f.write("\n")

    f.write("ORPHAN TRANSCRIPTS REMOVED\n")
    f.write("-" * 60 + "\n")

    for item in orphan_transcripts:
        f.write(f"{item}\n")

    f.write("\n")

    f.write("ORPHAN AUDIO REMOVED\n")
    f.write("-" * 60 + "\n")

    for item in orphan_audio:
        f.write(f"{item}\n")

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print(f"Bad annotation sets removed : {len(bad_ids)}")
print(f"Annotations deleted         : {deleted_annotations}")
print(f"Transcripts deleted         : {deleted_transcripts}")
print(f"Audio deleted               : {deleted_audio}")

print(f"\nReport saved to: {REPORT_FILE}")

print("=" * 60)
