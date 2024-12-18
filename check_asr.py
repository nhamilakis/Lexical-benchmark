import sys
from pathlib import Path

import pygame
from PyQt6.QtWidgets import QApplication, QComboBox, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget


class SegmentPlayer(QMainWindow):
    def __init__(self, segments_dir: Path):
        super().__init__()
        pygame.mixer.init()
        self.segments_dir = segments_dir
        self.segments = self.load_segments()
        self.setup_ui()

    def load_segments(self) -> dict[str, dict[str, Path]]:
        segments = {}
        for wav_file in self.segments_dir.glob("seg*.wav"):
            txt_file = wav_file.with_suffix(".txt")
            if txt_file.exists():
                segment_name = wav_file.stem
                segments[segment_name] = {"audio": wav_file, "text": txt_file}
        return segments

    def setup_ui(self) -> None:
        self.setWindowTitle("Segment Player")
        main_widget = QWidget()
        layout = QVBoxLayout()

        # Segment selector
        self.segment_combo = QComboBox()
        self.segment_combo.addItems(sorted(self.segments.keys()))
        self.segment_combo.currentTextChanged.connect(self.on_segment_change)
        layout.addWidget(self.segment_combo)

        # Transcription display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        layout.addWidget(self.text_display)

        # Control buttons
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play)
        layout.addWidget(self.play_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        layout.addWidget(self.stop_button)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.on_segment_change(self.segment_combo.currentText())

    def on_segment_change(self, segment: str) -> None:
        if segment in self.segments:
            text_path = self.segments[segment]["text"]
            self.text_display.setText(text_path.read_text())

    def play(self) -> None:
        segment = self.segment_combo.currentText()
        if segment in self.segments:
            pygame.mixer.music.load(str(self.segments[segment]["audio"]))
            pygame.mixer.music.play()

    def stop(self) -> None:
        pygame.mixer.music.stop()


def main():
    app = QApplication(sys.argv)
    player = SegmentPlayer(Path("data/asr"))
    player.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
