import mir_eval
import numpy as np
import constants
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf
import pretty_midi

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

    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals,
                                                                                 pretty_midi.note_number_to_hz(ref_pitches), 
                                                                                est_intervals, 
                                                                                pretty_midi.note_number_to_hz(est_pitches),
                                                                                 offset_ratio=None)

    return precision, recall, f1

def get_f1_score_frames(ref_frames, est_frames):
    precision, recall, f1, _ = precision_recall_fscore_support(ref_frames.flatten(), est_frames.flatten())
    return precision, recall, f1

