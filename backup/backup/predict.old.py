import asyncio
import contextlib
import json
import time
from pathlib import Path

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import pygame
import soundfile as sf
import torch
from config import AUDIO_DIR, DEVICE, DTYPE_RECORD, MODEL_OUTPUT_DIR, NOISE_FILENAME, SAMPLE_RATE, WINDOW_TIME
from evdev import InputDevice, categorize, ecodes
from record import AudioRecorder
from train import KeystrokeCNN
from utils import compute_log_mel_spectrogram

mpl.use("TkAgg")

# --- Pygame and Keyboard UI Configuration ---
raw_dir = AUDIO_DIR / "raw"
n, sr = sf.read(raw_dir / NOISE_FILENAME, dtype=DTYPE_RECORD)
_ = librosa.resample(y=n.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
print("Librosa is working")

pygame.init()

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
LIGHT_GREY = (200, 200, 200)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

# --- Fonts ---
FONT_LARGE = pygame.font.Font(None, 48)
FONT_MEDIUM = pygame.font.Font(None, 36)
FONT_SMALL = pygame.font.Font(None, 24)

# --- Screen ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Keystroke Prediction")

# --- Keyboard Layout (QWERTY) ---
# Note: This is a simplified mapping. For a full mapping, a more complex dictionary would be needed.
KEY_LAYOUT = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "0": "0",
    "-": "-",
    "=": "=",
    "q": "q",
    "w": "w",
    "e": "e",
    "r": "r",
    "t": "t",
    "y": "y",
    "u": "u",
    "i": "i",
    "o": "o",
    "p": "p",
    "[": "[",
    "]": "]",
    "a": "a",
    "s": "s",
    "d": "d",
    "f": "f",
    "g": "g",
    "h": "h",
    "j": "j",
    "k": "k",
    "l": "l",
    ";": ";",
    "'": "'",
    "z": "z",
    "x": "x",
    "c": "c",
    "v": "v",
    "b": "b",
    "n": "n",
    "m": "m",
    ",": ",",
    ".": ".",
    "/": "/",
    "space": " ",
}
KEY_VISUAL_LAYOUT = [
    "1234567890-=",
    "qwertyuiop[]",
    "asdfghjkl;'",
    "zxcvbnm,./",
]
KEY_WIDTH = 60
KEY_HEIGHT = 60
KEY_MARGIN = 10
MID_CONFIDENCE_THRESHOLD = 0.5
MIN_VISIBLE_HEATMAP_CONFIDENCE = 0.1


def plot_time(signal: np.ndarray, name: str):
    """Plot the signal in the time domain."""
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(f"Time Domain Signal: {name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(AUDIO_DIR / f"images/{name}.png")


class Keyboard:
    def __init__(self):
        self.keys = {}
        self.key_surfaces = {}
        y_offset = 100
        for row_idx, row in enumerate(KEY_VISUAL_LAYOUT):
            x_offset = (SCREEN_WIDTH - (len(row) * (KEY_WIDTH + KEY_MARGIN))) // 2
            for char_idx, char in enumerate(row):
                x = x_offset + char_idx * (KEY_WIDTH + KEY_MARGIN)
                y = y_offset + row_idx * (KEY_HEIGHT + KEY_MARGIN)
                self.keys[char] = pygame.Rect(x, y, KEY_WIDTH, KEY_HEIGHT)
        # Add space bar
        space_y = y_offset + len(KEY_VISUAL_LAYOUT) * (KEY_HEIGHT + KEY_MARGIN)
        space_width = 7 * (KEY_WIDTH + KEY_MARGIN)
        space_x = (SCREEN_WIDTH - space_width) // 2
        self.keys["space"] = pygame.Rect(space_x, space_y, space_width, KEY_HEIGHT)

    def create_key_surface(self, char, color, rect):
        key_surface = pygame.Surface(rect.size)
        key_surface.fill(color)
        text_surface = FONT_MEDIUM.render(char, True, BLACK)
        text_rect = text_surface.get_rect(center=(rect.width // 2, rect.height // 2))
        key_surface.blit(text_surface, text_rect)
        return key_surface

    def draw(self, pressed_key=None, heatmap=None):
        for char, rect in self.keys.items():
            display_char = char if char != "space" else "SPACE"
            color = WHITE
            if pressed_key and char == pressed_key:
                color = GREY
            elif heatmap and char in heatmap:
                confidence = heatmap[char]
                color = self.get_heatmap_color(confidence)

            key_surface = self.create_key_surface(display_char, color, rect)
            screen.blit(key_surface, rect.topleft)

    def get_heatmap_color(self, confidence: float) -> pygame.Color:
        # Ensure confidence is within [0, 1] range
        confidence = max(0.0, min(1.0, confidence))

        if confidence >= MID_CONFIDENCE_THRESHOLD:
            lerp_factor = (confidence - MID_CONFIDENCE_THRESHOLD) / (1.0 - MID_CONFIDENCE_THRESHOLD)
            return pygame.Color(YELLOW).lerp(pygame.Color(GREEN), lerp_factor)

        if MID_CONFIDENCE_THRESHOLD == 0:  # Avoid division by zero if threshold is 0
            return pygame.Color(WHITE) if confidence == 0 else pygame.Color(YELLOW)
        lerp_factor = confidence / MID_CONFIDENCE_THRESHOLD
        return pygame.Color(WHITE).lerp(pygame.Color(YELLOW), lerp_factor)


class Predictor:
    def __init__(self):
        self.model, self.metadata = self.load_model_and_metadata()
        self.key_map = {v: k for k, v in self.metadata["key_map"].items()}
        self.noise_data = self.load_noise_data()
        self.recorder = AudioRecorder("", 0)
        self.processing_lock = False
        self.is_recording = False
        self.last_key_time = None
        self.keystroke_events = []
        self.predicted_sentences = []
        self.selected_sentence_idx = 0
        self.selected_key_idx = -1

    def load_model_and_metadata(self):
        MODEL_PATH = MODEL_OUTPUT_DIR / "keystroke_classifier.pth"
        METADATA_PATH = MODEL_OUTPUT_DIR / "model_metadata.json"
        with Path.open(METADATA_PATH) as f:
            metadata = json.load(f)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model = KeystrokeCNN(
            n_mels=metadata["n_mels"],
            time_bins=metadata["time_bins"],
            num_classes=metadata["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        return model, metadata

    def load_noise_data(self):
        raw_dir = AUDIO_DIR / "raw"
        noise_data, sr = sf.read(raw_dir / NOISE_FILENAME, dtype=DTYPE_RECORD)
        return librosa.resample(
            y=noise_data.astype(np.float32),
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
        )

    async def start_recording(self):
        if not self.is_recording:
            self.recorder = AudioRecorder("", 0)
            self.recorder.start()
            self.keystroke_events = []
            print("Recorder initializing, please wait...")
            await asyncio.sleep(0.3)  # Wait for recorder to stabilize
            self.is_recording = True
            print("Recording started...")

    def stop_and_process(self):
        if self.is_recording:
            print("Stopping recording and processing...")
            self.show_processing_bar()
            y, sr = self.recorder.stop()
            self.is_recording = False
            self.predict_keystrokes(y, sr)

    def show_processing_bar(self):
        loading_text = FONT_MEDIUM.render("Processing...", True, WHITE)
        loading_rect = loading_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        screen.blit(loading_text, loading_rect)
        pygame.display.flip()

    def predict_keystrokes(self, y, sr):
        if len(y) == 0 or not self.keystroke_events:
            return

        y_resampled = librosa.resample(
            y=y.astype(np.float32),
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
        )

        # if self.noise_data.dtype != np.float32:
        #     self.noise_data = librosa.resample(
        #         y=self.noise_data.astype(np.float32),
        #         orig_sr=sr,
        #         target_sr=SAMPLE_RATE,
        #     )

        y_clean = nr.reduce_noise(y=y_resampled, sr=SAMPLE_RATE, y_noise=self.noise_data)
        if np.max(np.abs(y_clean)) > 0:
            y_clean /= np.max(np.abs(y_clean))

        window = int(WINDOW_TIME * SAMPLE_RATE)
        all_sentence_probs = []

        i = 0
        for event_time in self.keystroke_events:
            t = event_time - (self.recorder.start_time - 0.025)
            center_sample = int(t * SAMPLE_RATE)
            start = max(0, center_sample - window // 2)
            end = min(len(y_clean), start + window)
            snippet = y_clean[start:end]

            if snippet.shape[0] < window:
                snippet = np.pad(snippet, (0, window - snippet.shape[0]), "constant")

            plot_time(snippet, f"{i}")

            _, _, all_probs = self.predict_keystroke(snippet)
            all_sentence_probs.append(all_probs)
            i += 1

        # Determine the single best sentence by picking the highest probability key at each step
        best_sentence_text = ""
        sentence_heatmaps_probs = []  # List of raw probability arrays for each keystroke

        for probs_for_keystroke in all_sentence_probs:  # all_sentence_probs is a list of numpy arrays
            best_key_idx = np.argmax(probs_for_keystroke)
            # self.key_map maps class index to character string (e.g., 'a', 'b', ..., 'space')
            key_char_from_model = self.key_map.get(best_key_idx, "?")

            # For display text, use " " for space.
            # The key_char_from_model (e.g. "space") is used later for heatmap keys.
            display_char = " " if key_char_from_model == "space" else key_char_from_model
            best_sentence_text += display_char
            sentence_heatmaps_probs.append(probs_for_keystroke)

        if best_sentence_text:
            self.predicted_sentences = [{"text": best_sentence_text, "heatmaps": sentence_heatmaps_probs}]
        else:
            self.predicted_sentences = []

        self.selected_key_idx = -1  # Reset to the beginning of the new sentence

    def predict_keystroke(self, signal):
        mel_spec = compute_log_mel_spectrogram(signal)
        input_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        predicted_key = self.key_map.get(predicted_class, "?")
        return predicted_key, confidence, probabilities[0].cpu().numpy()

    def handle_keypress_event(self, key_name):
        if key_name == "left":
            if self.selected_key_idx > 0:
                self.selected_key_idx -= 1
            elif self.selected_key_idx == 0:
                self.selected_key_idx = -1  # Go to aggregate view
            # If self.selected_key_idx is already -1, it stays -1
        elif key_name == "right":
            if self.predicted_sentences and self.predicted_sentences[0]["text"]:
                sentence_text = self.predicted_sentences[0]["text"]
                sentence_len = len(sentence_text)
                if sentence_len > 0:
                    if self.selected_key_idx == -1:
                        self.selected_key_idx = 0  # Move from aggregate to first char
                    elif self.selected_key_idx < sentence_len - 1:
                        self.selected_key_idx += 1

    def handle_key_press(self):
        if self.is_recording:
            self.last_key_time = time.time()
            self.keystroke_events.append(self.last_key_time)

    async def update(self):
        if not self.is_recording and not self.processing_lock:
            await self.start_recording()

        if self.is_recording and self.last_key_time and (time.time() - self.last_key_time > 2.0):
            self.processing_lock = True
            self.stop_and_process()
            self.last_key_time = None
            self.processing_lock = False

    def draw_predictions(self):
        y_offset = SCREEN_HEIGHT // 2 + 50  # Adjusted offset for better layout
        if self.predicted_sentences:
            sentence_data = self.predicted_sentences[0]
            text_to_display = sentence_data["text"]

            # Create a surface for the text to handle character-by-character coloring
            # Ensure the surface is wide enough. Max width or calculate based on text.
            text_render_area_width = SCREEN_WIDTH - 100  # Example width
            text_render_area_height = FONT_MEDIUM.get_height()

            full_text_surface = pygame.Surface((text_render_area_width, text_render_area_height), pygame.SRCALPHA)
            full_text_surface.fill((0, 0, 0, 0))  # Transparent background

            current_x = 0
            for i, char_in_sentence in enumerate(text_to_display):
                color = WHITE
                if i == self.selected_key_idx and self.selected_key_idx != -1:
                    color = YELLOW  # Highlight the selected character

                char_surface = FONT_MEDIUM.render(char_in_sentence, True, color)
                full_text_surface.blit(char_surface, (current_x, 0))
                current_x += char_surface.get_width()

            screen.blit(full_text_surface, (50, y_offset))

    def get_current_heatmap(self):
        if not self.predicted_sentences:
            return None

        current_prediction = self.predicted_sentences[0]
        sentence_text = current_prediction["text"]
        sentence_heatmaps = current_prediction["heatmaps"]  # This is a list of probability arrays

        if not sentence_text:  # Or if sentence_heatmaps is empty
            return None

        heatmap_to_display = {}

        if self.selected_key_idx == -1:  # Aggregate heatmap
            # For aggregated view, take the maximum probability for each key across all positions
            aggregated_probs = {}
            for probs_for_char_position in sentence_heatmaps:
                for class_idx, probability in enumerate(probs_for_char_position):
                    key_identifier = self.key_map.get(class_idx, None)
                    if key_identifier and (key_identifier in KEY_LAYOUT or key_identifier == "space"):
                        current_max_prob = aggregated_probs.get(key_identifier, 0.0)
                        aggregated_probs[key_identifier] = max(current_max_prob, probability)
            heatmap_to_display = aggregated_probs
        elif self.selected_key_idx >= 0 and self.selected_key_idx < len(sentence_heatmaps):
            # Get the probability distribution for the currently selected keystroke/character
            probs_for_selected_char_position = sentence_heatmaps[self.selected_key_idx]
            for class_idx, probability in enumerate(probs_for_selected_char_position):
                key_identifier = self.key_map.get(class_idx, None)
                if key_identifier and (key_identifier in KEY_LAYOUT or key_identifier == "space"):
                    heatmap_to_display[key_identifier] = probability
        else:
            # This case should ideally not be reached if selected_key_idx is managed correctly
            return None

        return heatmap_to_display


async def key_event_producer(device, queue):
    """Listens for keyboard events and puts them into the queue."""
    async for event in device.async_read_loop():
        await queue.put(event)


async def main():
    predictor = Predictor()
    keyboard = Keyboard()
    clock = pygame.time.Clock()
    running = True
    pressed_key_char = None
    key_pressed_time = 0

    try:
        dev = InputDevice("/dev/input/event0")
        dev.grab()
    except (PermissionError, FileNotFoundError) as e:
        print(f"Error initializing input device: {e}")
        print("Please ensure you are running with sufficient permissions (e.g., sudo) and the device path is correct.")
        return

    event_queue = asyncio.Queue()
    producer_task = asyncio.create_task(key_event_producer(dev, event_queue))

    print("Application started. Type to begin.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False  # Allow Esc to quit

        try:
            while not event_queue.empty():
                ev = event_queue.get_nowait()
                if ev.type == ecodes.EV_KEY and ev.value == 1:  # Key press
                    key = categorize(ev)
                    key_name = key.keycode.replace("KEY_", "").lower()

                    if key_name == "esc":
                        running = False

                    predictor.handle_keypress_event(key_name)

                    if key_name in KEY_LAYOUT:
                        predictor.handle_key_press()
                        pressed_key_char = key_name
                        key_pressed_time = time.time()
        except asyncio.QueueEmpty:
            pass

        await predictor.update()

        if pressed_key_char and (time.time() - key_pressed_time > 0.2):
            pressed_key_char = None

        screen.fill(BLACK)
        heatmap = predictor.get_current_heatmap()
        keyboard.draw(pressed_key_char, heatmap)
        predictor.draw_predictions()

        pygame.display.flip()

        await asyncio.sleep(1 / 30)  # Limit FPS and allow other tasks to run
        clock.tick(30)

    producer_task.cancel()
    dev.ungrab()
    pygame.quit()
    print("Exiting program.")


if __name__ == "__main__":
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(main())
