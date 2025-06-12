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

# --- Constants for prediction behavior ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

RECORDING_AUTO_STOP_DELAY = 2.0
KEY_VISUAL_CUE_DURATION = 0.2
PREDICTION_TABLE_MAX_ITEMS = 5
PREDICTION_TABLE_X_OFFSET = 50
PREDICTION_TABLE_Y_OFFSET = 600
PREDICTION_DISPLAY_Y = SCREEN_HEIGHT // 2 + 30  # Y position for the CENTER prediction line
INTER_LINE_SPACING = 5  # Spacing between prediction lines

# --- Pygame and Keyboard UI Configuration ---
raw_dir = AUDIO_DIR / "raw"
n, sr_file = sf.read(raw_dir / NOISE_FILENAME, dtype=DTYPE_RECORD)
_ = librosa.resample(y=n.astype(np.float32), orig_sr=sr_file, target_sr=SAMPLE_RATE)
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
FONT_MEDIUM = pygame.font.Font(None, 36)  # For center prediction line
FONT_SMALL = pygame.font.Font(None, 28)  # For edge prediction lines

# --- Screen ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Keystroke Prediction")

# --- Keyboard Layout (QWERTY) ---
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
        y_offset = 100
        for row_idx, row in enumerate(KEY_VISUAL_LAYOUT):
            x_offset = (SCREEN_WIDTH - (len(row) * (KEY_WIDTH + KEY_MARGIN))) // 2
            for char_idx, char in enumerate(row):
                x = x_offset + char_idx * (KEY_WIDTH + KEY_MARGIN)
                y = y_offset + row_idx * (KEY_HEIGHT + KEY_MARGIN)
                self.keys[char] = pygame.Rect(x, y, KEY_WIDTH, KEY_HEIGHT)
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
                color = self.get_heatmap_color(heatmap[char])
            key_surface = self.create_key_surface(display_char, color, rect)
            screen.blit(key_surface, rect.topleft)

    def get_heatmap_color(self, confidence: float) -> pygame.Color:
        confidence = max(0.0, min(1.0, confidence))
        if confidence >= MID_CONFIDENCE_THRESHOLD:
            lerp_factor = (
                (confidence - MID_CONFIDENCE_THRESHOLD) / (1.0 - MID_CONFIDENCE_THRESHOLD)
                if (1.0 - MID_CONFIDENCE_THRESHOLD) > 0
                else 1.0
            )
            return pygame.Color(YELLOW).lerp(pygame.Color(GREEN), lerp_factor)
        if MID_CONFIDENCE_THRESHOLD == 0:
            return pygame.Color(WHITE) if confidence == 0 else pygame.Color(YELLOW)
        lerp_factor = confidence / MID_CONFIDENCE_THRESHOLD if MID_CONFIDENCE_THRESHOLD > 0 else 1.0
        return pygame.Color(WHITE).lerp(pygame.Color(YELLOW), lerp_factor)


class Predictor:
    PREDICTION_HYPOTHESES_COUNT = 30
    FADE_DURATION = 0.1  # Seconds for the fade effect (slightly faster)

    def __init__(self):
        self.model, self.metadata = self.load_model_and_metadata()
        self.key_map = {v: k for k, v in self.metadata["key_map"].items()}
        self.noise_data = self.load_noise_data()
        self.recorder = AudioRecorder("", 0)
        self.processing_lock = False
        self.is_recording = False
        self.last_key_time = None
        self.keystroke_events = []
        self.predicted_sentences_details = []
        self.selected_sentence_idx = 0
        self.selected_key_idx = -1

        self.is_fading = False
        self.fade_progress = 0.0
        self.fade_start_time = 0.0
        self.surface_to_fade_out = None
        self.surface_to_fade_in = None
        self.current_sentence_surface = None  # For the center (selected) line

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
        noise_data, sr_noise = sf.read(raw_dir / NOISE_FILENAME, dtype=DTYPE_RECORD)
        return librosa.resample(y=noise_data.astype(np.float32), orig_sr=sr_noise, target_sr=SAMPLE_RATE)

    async def start_recording(self):
        if not self.is_recording:
            self.recorder = AudioRecorder("", 0)
            self.recorder.start()
            self.keystroke_events = []
            print("Recorder initializing, please wait...")
            await asyncio.sleep(0.3)
            self.is_recording = True
            print("Recording started...")

    def stop_and_process(self):
        if self.is_recording:
            print("Stopping recording and processing...")
            self.show_processing_bar()
            y, sr_rec = self.recorder.stop()
            self.is_recording = False
            self.predict_keystrokes(y, sr_rec)

    def show_processing_bar(self):
        loading_text = FONT_MEDIUM.render("Processing...", True, WHITE)
        loading_rect = loading_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        screen.blit(loading_text, loading_rect)
        pygame.display.flip()

    def predict_keystrokes(self, y, sr_rec):
        if len(y) == 0 or not self.keystroke_events:
            self.predicted_sentences_details = []
        else:
            y_resampled = librosa.resample(y=y.astype(np.float32), orig_sr=sr_rec, target_sr=SAMPLE_RATE)
            y_clean = nr.reduce_noise(y=y_resampled, sr=SAMPLE_RATE, y_noise=self.noise_data)
            if np.max(np.abs(y_clean)) > 0:
                y_clean /= np.max(np.abs(y_clean))
            window = int(WINDOW_TIME * SAMPLE_RATE)
            per_position_data = []
            for i, event_data in enumerate(self.keystroke_events):
                event_time, actual_key_char = event_data["time"], event_data["key_char"]
                t = event_time - (self.recorder.start_time - 0.025)
                center_sample = int(t * SAMPLE_RATE)
                start = max(0, center_sample - window // 2)
                end = min(len(y_clean), start + window)
                snippet = y_clean[start:end]
                if snippet.shape[0] < window:
                    snippet = np.pad(snippet, (0, window - snippet.shape[0]), "constant")
                mel_spec = compute_log_mel_spectrogram(snippet)
                input_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(logits, dim=1)
                per_position_data.append({"actual_char": actual_key_char, "all_probs": probabilities[0].cpu().numpy()})
            self.predicted_sentences_details = (
                self.generate_sentence_hypotheses(per_position_data) if per_position_data else []
            )
        self.keystroke_events = []
        self.selected_sentence_idx = 0
        self.selected_key_idx = -1
        self.is_fading = False
        self.surface_to_fade_out = None
        self.surface_to_fade_in = None
        if self.predicted_sentences_details:
            self.current_sentence_surface = self._render_center_sentence_surface(self.predicted_sentences_details[0], 1)
        else:
            self.current_sentence_surface = None

    def generate_sentence_hypotheses(self, per_position_data):
        if not per_position_data:
            return []
        num_positions = len(per_position_data)
        final_hypotheses = []
        source_probs_list = [item["all_probs"] for item in per_position_data]
        source_actual_chars_list = [item["actual_char"] for item in per_position_data]
        greedy_chars = [self.key_map.get(np.argmax(probs), "?") for probs in source_probs_list]
        greedy_log_score = sum(
            np.log(probs[np.argmax(probs)]) if probs[np.argmax(probs)] > 1e-9 else np.log(1e-9)
            for probs in source_probs_list
        )
        final_hypotheses.append(
            {
                "id": time.time(),
                "sequence_actual_chars": list(source_actual_chars_list),
                "sequence_predicted_chars": greedy_chars,
                "sequence_score": np.exp(greedy_log_score),
                "per_char_probs": list(source_probs_list),
            },
        )
        if self.PREDICTION_HYPOTHESES_COUNT <= 1 or num_positions == 0:
            return final_hypotheses
        beams = [([], 0.0)]
        for pos_idx in range(num_positions):
            all_probs_at_pos = source_probs_list[pos_idx]
            new_beams = []
            for seq_indices, current_log_score in beams:
                for char_class_idx, prob in enumerate(all_probs_at_pos):
                    if prob > 1e-9:
                        new_beams.append((seq_indices + [char_class_idx], current_log_score + np.log(prob)))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[: self.PREDICTION_HYPOTHESES_COUNT]
        added_sequences_str = {"".join(greedy_chars)}
        for seq_indices, total_log_score in beams:
            if len(final_hypotheses) >= self.PREDICTION_HYPOTHESES_COUNT:
                break
            predicted_chars = [self.key_map.get(idx, "?") for idx in seq_indices]
            current_sequence_str = "".join(predicted_chars)
            if current_sequence_str in added_sequences_str:
                continue
            final_hypotheses.append(
                {
                    "id": time.time() + len(final_hypotheses) * 0.001,
                    "sequence_actual_chars": list(source_actual_chars_list),
                    "sequence_predicted_chars": predicted_chars,
                    "sequence_score": np.exp(total_log_score),
                    "per_char_probs": list(source_probs_list),
                },
            )
            added_sequences_str.add(current_sequence_str)
        return final_hypotheses

    def _render_center_sentence_surface(self, hypothesis, display_number):  # For the main, selected line
        if not hypothesis:
            return None
        predicted_chars, actual_chars = hypothesis["sequence_predicted_chars"], hypothesis["sequence_actual_chars"]
        surface_width = SCREEN_WIDTH - 100
        surface_height = FONT_MEDIUM.get_height()
        sentence_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)
        sentence_surface.fill((0, 0, 0, 0))
        prefix_text = f"{display_number}. "
        prefix_surface = FONT_MEDIUM.render(prefix_text, True, YELLOW)
        sentence_surface.blit(prefix_surface, (0, 0))
        current_x = prefix_surface.get_width()
        for char_idx, pred_char in enumerate(predicted_chars):
            actual_char = actual_chars[char_idx]
            char_disp = " " if pred_char == "space" else pred_char
            char_color = GREEN if pred_char == actual_char else RED
            if char_idx == self.selected_key_idx:
                char_color = YELLOW
            char_surf = FONT_MEDIUM.render(char_disp, True, char_color)
            if current_x + char_surf.get_width() <= surface_width:
                sentence_surface.blit(char_surf, (current_x, 0))
                current_x += char_surf.get_width()
            else:
                break
        return sentence_surface

    def _render_edge_sentence_surface(self, hypothesis, display_number):  # For top/bottom lines
        if not hypothesis:
            return None
        predicted_chars = hypothesis["sequence_predicted_chars"]
        surface_width = SCREEN_WIDTH - 100
        surface_height = FONT_SMALL.get_height()
        sentence_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)
        sentence_surface.fill((0, 0, 0, 0))
        prefix_text = f"{display_number}. "
        full_text = prefix_text + "".join(" " if p == "space" else p for p in predicted_chars)
        text_surf = FONT_SMALL.render(full_text, True, GREY)  # Render whole line in GREY
        sentence_surface.set_alpha(150)  # Reduced opacity for edge lines
        sentence_surface.blit(text_surf, (0, 0))
        return sentence_surface

    def _start_fade(self, old_idx, new_idx):
        if not self.predicted_sentences_details or old_idx == new_idx:
            return
        if self.current_sentence_surface and old_idx == (self.selected_sentence_idx if not self.is_fading else -1):
            self.surface_to_fade_out = self.current_sentence_surface.copy()
        elif 0 <= old_idx < len(self.predicted_sentences_details):
            self.surface_to_fade_out = self._render_center_sentence_surface(
                self.predicted_sentences_details[old_idx],
                old_idx + 1,
            )
        else:
            self.surface_to_fade_out = None
        self.surface_to_fade_in = self._render_center_sentence_surface(
            self.predicted_sentences_details[new_idx],
            new_idx + 1,
        )
        if self.surface_to_fade_in:
            self.is_fading = True
            self.fade_progress = 0.0
            self.fade_start_time = time.time()
        else:
            self.is_fading = False
            self.current_sentence_surface = self.surface_to_fade_out
            self.surface_to_fade_out = None

    def select_previous_hypothesis(self):
        if not self.predicted_sentences_details or self.is_fading:
            return
        num_hyp = len(self.predicted_sentences_details)
        old_idx = self.selected_sentence_idx
        self.selected_sentence_idx = (self.selected_sentence_idx - 1 + num_hyp) % num_hyp
        self.selected_key_idx = -1
        if old_idx != self.selected_sentence_idx:
            self._start_fade(old_idx, self.selected_sentence_idx)

    def select_next_hypothesis(self):
        if not self.predicted_sentences_details or self.is_fading:
            return
        num_hyp = len(self.predicted_sentences_details)
        old_idx = self.selected_sentence_idx
        self.selected_sentence_idx = (self.selected_sentence_idx + 1) % num_hyp
        self.selected_key_idx = -1
        if old_idx != self.selected_sentence_idx:
            self._start_fade(old_idx, self.selected_sentence_idx)

    def handle_keypress_event(self, key_name):
        if key_name == "left":
            if self.selected_key_idx >= 0:
                self.selected_key_idx -= 1
            if self.predicted_sentences_details and not self.is_fading:  # Re-render center line
                self.current_sentence_surface = self._render_center_sentence_surface(
                    self.predicted_sentences_details[self.selected_sentence_idx],
                    self.selected_sentence_idx + 1,
                )
        elif key_name == "right":
            if self.predicted_sentences_details and self.selected_sentence_idx < len(self.predicted_sentences_details):
                hyp = self.predicted_sentences_details[self.selected_sentence_idx]
                sentence_len = len(hyp["sequence_predicted_chars"])
                if sentence_len > 0:
                    if self.selected_key_idx == -1:
                        self.selected_key_idx = 0
                    elif self.selected_key_idx < sentence_len - 1:
                        self.selected_key_idx += 1
            if self.predicted_sentences_details and not self.is_fading:  # Re-render center line
                self.current_sentence_surface = self._render_center_sentence_surface(
                    self.predicted_sentences_details[self.selected_sentence_idx],
                    self.selected_sentence_idx + 1,
                )
        elif key_name == "up":
            self.select_previous_hypothesis()
        elif key_name == "down":
            self.select_next_hypothesis()

    def handle_key_press(self, key_name_pressed: str):
        if self.is_recording:
            self.last_key_time = time.time()
            self.keystroke_events.append({"time": self.last_key_time, "key_char": key_name_pressed})

    async def update(self):
        if not self.is_recording and not self.processing_lock:
            await self.start_recording()
        if self.is_recording and self.last_key_time and (time.time() - self.last_key_time > RECORDING_AUTO_STOP_DELAY):
            self.processing_lock = True
            self.stop_and_process()
            self.last_key_time = None
            self.processing_lock = False

    def draw_predictions(self):
        if not self.predicted_sentences_details:
            return
        num_hypotheses = len(self.predicted_sentences_details)

        center_y = PREDICTION_DISPLAY_Y
        top_y = center_y - FONT_SMALL.get_height() - INTER_LINE_SPACING
        bottom_y = center_y + FONT_MEDIUM.get_height() + INTER_LINE_SPACING

        # --- Center Line (Selected) ---
        if self.is_fading:
            elapsed_time = time.time() - self.fade_start_time
            self.fade_progress = min(1.0, elapsed_time / self.FADE_DURATION)
            alpha_out = int(255 * (1.0 - self.fade_progress))
            alpha_in = int(255 * self.fade_progress)
            if self.surface_to_fade_out:
                self.surface_to_fade_out.set_alpha(alpha_out)
                screen.blit(self.surface_to_fade_out, (50, center_y))
            if self.surface_to_fade_in:
                self.surface_to_fade_in.set_alpha(alpha_in)
                screen.blit(self.surface_to_fade_in, (50, center_y))
            if self.fade_progress >= 1.0:
                self.is_fading = False
                self.current_sentence_surface = self.surface_to_fade_in
                self.surface_to_fade_out = None
                self.surface_to_fade_in = None
        elif self.current_sentence_surface:
            self.current_sentence_surface.set_alpha(255)
            screen.blit(self.current_sentence_surface, (50, center_y))

        # --- Top Line (Previous Hypothesis) ---
        top_item_idx = (self.selected_sentence_idx - 1 + num_hypotheses) % num_hypotheses
        if num_hypotheses >= 2 and top_item_idx != self.selected_sentence_idx:
            hypothesis = self.predicted_sentences_details[top_item_idx]
            surface = self._render_edge_sentence_surface(hypothesis, top_item_idx + 1)
            if surface:
                screen.blit(surface, (50, top_y))

        # --- Bottom Line (Next Hypothesis) ---
        bottom_item_idx = (self.selected_sentence_idx + 1) % num_hypotheses
        if num_hypotheses >= 2 and bottom_item_idx != self.selected_sentence_idx:
            if not (
                num_hypotheses == 2 and bottom_item_idx == top_item_idx
            ):  # Avoid drawing same item twice if only 2 hypotheses
                hypothesis = self.predicted_sentences_details[bottom_item_idx]
                surface = self._render_edge_sentence_surface(hypothesis, bottom_item_idx + 1)
                if surface:
                    screen.blit(surface, (50, bottom_y))

    def get_current_heatmap(self):
        if not self.predicted_sentences_details or self.selected_sentence_idx >= len(self.predicted_sentences_details):
            return None
        current_hypothesis = self.predicted_sentences_details[self.selected_sentence_idx]
        all_probs = current_hypothesis["per_char_probs"]
        if not all_probs:
            return None
            heatmap = {}
        if self.selected_key_idx == -1:
            agg_probs = {}
            for probs_pos in all_probs:
                for class_idx, prob_val in enumerate(probs_pos):
                    key_id = self.key_map.get(class_idx)
                    if key_id and (key_id in KEY_LAYOUT or key_id == "space"):
                        agg_probs[key_id] = max(agg_probs.get(key_id, 0.0), float(prob_val))
            heatmap = agg_probs
        elif 0 <= self.selected_key_idx < len(all_probs):
            probs_sel_pos = all_probs[self.selected_key_idx]
            for class_idx, prob_val in enumerate(probs_sel_pos):
                key_id = self.key_map.get(class_idx)
                if key_id and (key_id in KEY_LAYOUT or key_id == "space"):
                    heatmap[key_id] = float(prob_val)
        return heatmap if heatmap else None

    def draw_prediction_table(self):
        if (
            not self.predicted_sentences_details
            or self.selected_sentence_idx >= len(self.predicted_sentences_details)
            or self.selected_key_idx == -1
        ):
            return
        hyp = self.predicted_sentences_details[self.selected_sentence_idx]
        all_probs, actual_chars, pred_chars = (
            hyp["per_char_probs"],
            hyp["sequence_actual_chars"],
            hyp["sequence_predicted_chars"],
        )
        if not (0 <= self.selected_key_idx < len(all_probs)):
            return
        probs_arr, actual_c, pred_c = (
            all_probs[self.selected_key_idx],
            actual_chars[self.selected_key_idx],
            pred_chars[self.selected_key_idx],
        )
        char_probs = sorted(
            [(self.key_map.get(i, "?"), p) for i, p in enumerate(probs_arr)],
            key=lambda x: x[1],
            reverse=True,
        )
        title = f"Probabilities for '{'SPACE' if pred_c == 'space' else pred_c.upper()}' (Actual: '{'SPACE' if actual_c == 'space' else actual_c.upper()}')"
        title_surf = FONT_SMALL.render(title, True, WHITE)
        screen.blit(title_surf, (PREDICTION_TABLE_X_OFFSET, PREDICTION_TABLE_Y_OFFSET))
        lh = FONT_SMALL.get_height() + 2
        for i, (char, prob) in enumerate(char_probs[:PREDICTION_TABLE_MAX_ITEMS]):
            disp_char = "SPACE" if char == "space" else char.upper()
            text = f"- {disp_char}: {prob:.1%}"
            color = GREEN if char == actual_c else YELLOW if char == pred_c else WHITE
            text_surf = FONT_SMALL.render(text, True, color)
            screen.blit(
                text_surf,
                (PREDICTION_TABLE_X_OFFSET + 10, PREDICTION_TABLE_Y_OFFSET + title_surf.get_height() + 5 + i * lh),
            )


async def key_event_producer(device, queue):
    async for event in device.async_read_loop():
        await queue.put(event)


async def main():
    predictor = Predictor()
    keyboard = Keyboard()
    clock = pygame.time.Clock()
    running = True
    pressed_key_char, key_pressed_time = None, 0
    try:
        dev = InputDevice("/dev/input/event0")
        dev.grab()
    except (PermissionError, FileNotFoundError) as e:
        print(f"Error: {e}. Run with sudo & check device path.")
        return
    event_queue = asyncio.Queue()
    producer_task = asyncio.create_task(key_event_producer(dev, event_queue))
    print("Application started. Type to begin.")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        try:
            while not event_queue.empty():
                ev = event_queue.get_nowait()
                if ev.type == ecodes.EV_KEY and ev.value == 1:
                    key = categorize(ev)
                    key_name = key.keycode.replace("KEY_", "").lower()
                    if key_name == "esc":
                        running = False
                        break
                    predictor.handle_keypress_event(key_name)
                    if key_name in KEY_LAYOUT or key_name == "space":
                        rec_key = "space" if key_name == "space" else key_name
                        predictor.handle_key_press(rec_key)
                        pressed_key_char, key_pressed_time = rec_key, time.time()
        except asyncio.QueueEmpty:
            pass
        if not running:
            break
        await predictor.update()
        if pressed_key_char and (time.time() - key_pressed_time > KEY_VISUAL_CUE_DURATION):
            pressed_key_char = None
        screen.fill(BLACK)
        keyboard.draw(pressed_key_char, predictor.get_current_heatmap())
        predictor.draw_predictions()
        predictor.draw_prediction_table()
        pygame.display.flip()
        await asyncio.sleep(1 / 60)
        clock.tick(60)
    producer_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await producer_task
    dev.ungrab()
    pygame.quit()
    print("Exiting program.")


if __name__ == "__main__":
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(main())
