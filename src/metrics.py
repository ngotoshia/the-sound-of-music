import mir_eval
import numpy as np
import constants

def pm_to_pitches_intervals(pm,min_midi_pitch=constants.MIN_MIDI,
                                 max_midi_pitch=constants.MAX_MIDI):
    """Convert a NoteSequence to valued intervals."""
    intervals = []
    pitches = []
    velocities = []

    for note in pm.notes:
        if note.pitch < min_midi_pitch or note.pitch > max_midi_pitch:
            continue
        if note.end == note.start:
            continue
        intervals.append((note.start, note.end))
        pitches.append(note.pitch)
    return (np.array(intervals).reshape((-1, 2)), np.array(pitches))


def get_f1_score_notes(ref_intervals, ref_pitches, est_intervals, est_pitches):

    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches)

    return precision, recall, f1