# ruff: noqa
class NoteSequence:
    class Note:
        def __init__(self, pitch=0, start_time=0.0, end_time=0.0, velocity=80):
            self.pitch = pitch
            self.start_time = start_time
            self.end_time = end_time
            self.velocity = velocity

        def __repr__(self):
            return f"Note(pitch={self.pitch}, start={self.start_time})"
