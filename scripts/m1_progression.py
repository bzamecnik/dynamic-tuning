"""M1: generate the 12-key test progression, save it as MIDI and print the chord table."""

from pathlib import Path

from dynamic_tuning.progression import chord_sequence, chord_table, progression_score, voice_lead
from dynamic_tuning.score import write_midi

OUT = Path("outputs")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    chords = chord_sequence()
    voicing = voice_lead(chords)
    score = progression_score(chords, voicing)
    write_midi(score, OUT / "progression.mid")
    table = chord_table(chords, voicing)
    (OUT / "progression.txt").write_text(table + "\n")
    print(table)
    print(f"\n{len(score.notes)} notes, {score.duration:.1f} s, {len(score.sections)} sections")
    print(f"Wrote {OUT / 'progression.mid'}")


if __name__ == "__main__":
    main()
