from mido import MidiFile, MidiTrack, Message as MidiMessage


# creds to https://github.com/craffel/pretty-midi/issues/125
def piano_roll_to_midi(piano_roll, base_note=21):
    """Convert piano roll to a MIDI file."""
    notes, frames = piano_roll.shape
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    now = 0
    piano_roll = np.hstack((np.zeros((notes, 1)), 
                            piano_roll, 
                            np.zeros((notes, 1))))
    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        message = MidiMessage(
            type='note_on' if velocity > 0 else 'note_off', 
            note=int(note + base_note), 
            velocity=int(velocity * 127),
            time=int(time - now))
        track.append(message)
        now = time
    return midi