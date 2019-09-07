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

    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None)

    return precision, recall, f1

def get_f1_score_frames(ref_frames, est_frames):
    frame_true_positives = np.sum(np.logical_and(ref_frames==1, est_frames==1).astype(float))
    frame_false_positives = np.sum(np.logical_and(ref_frames==0, est_frames==1).astype(float))
    frame_false_negatives = np.sum(np.logical_and(ref_frames==1, est_frames==0).astype(float))

    precision = frame_true_positives/(frame_false_positives + frame_true_positives)
    recall = frame_true_positives/(frame_false_negatives + frame_true_positives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1