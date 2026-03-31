class MusicState:
    def __init__(self):
        self.tempo = 70
        self.harmonics = 2
        self.rhythm = "steady"
        self.intensity = 0.5

    def update(self, hrv, trend, velocity):
        if trend == "increasing":
            self.tempo += 2
            self.intensity += 0.05
        elif trend == "decreasing":
            self.tempo -= 3
            self.intensity -= 0.07
            self.rhythm = "irregular"
        elif trend == "stable":
            self.intensity -= 0.01

        if velocity > 5:
            self.tempo += 5
        if velocity < -5:
            self.rhythm = "very irregular"

        # Clamp values
        self.tempo = max(40, min(120, self.tempo))
        self.intensity = max(0.1, min(1.0, self.intensity))

        return {
            "tempo": self.tempo,
            "harmonics": self.harmonics,
            "rhythm": self.rhythm,
            "intensity": self.intensity,
        }
class MusicState:
    def __init__(self):
        self.tempo = 70
        self.harmonics = 2
        self.rhythm = "steady"
        self.intensity = 0.5

    def update(self, hrv, trend, velocity):
        if trend == "increasing":
            self.tempo += 2
            self.intensity += 0.05
        elif trend == "decreasing":
            self.tempo -= 3
            self.intensity -= 0.07
            self.rhythm = "irregular"
        elif trend == "stable":
            self.intensity -= 0.01

        if velocity > 5:
            self.tempo += 5
        if velocity < -5:
            self.rhythm = "very irregular"

        # Clamp values
        self.tempo = max(40, min(120, self.tempo))
        self.intensity = max(0.1, min(1.0, self.intensity))

        return {
            'tempo': self.tempo,
            'harmonics': self.harmonics,
            'rhythm': self.rhythm,
            'intensity': self.intensity
        }