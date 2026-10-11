import json
from pathlib import Path
import threading

import numpy as np
import pyaudio

import dearpygui.dearpygui as dpg

from models import (
    train_model,
    load_model,
    save_model,
    get_ear,
    get_audio_features,
    preprocess_data,
    predict,
    DEFAULT_MODEL_TYPE,
    MODEL_OPTIONS,
)
from bakeoff import (
    DEFAULT_CLASSES,
    TRIAL_KINDS,
    PredictionSmoother,
    evaluate_trial,
    format_results,
)

# Ensure metadata directory exists
Path("metadata/").mkdir(exist_ok=True)


# ——— Device Persistence Helpers ———


def save_metadata():
    with open("metadata/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def load_metadata():
    defaults = {
        "default_device": "",
        "data_classes": list(DEFAULT_CLASSES),
        "model_type": DEFAULT_MODEL_TYPE,
    }
    try:
        with open("metadata/metadata.json", "r") as f:
            return defaults | json.load(f)
    except FileNotFoundError:
        return defaults


metadata = load_metadata()
if metadata["model_type"] not in MODEL_OPTIONS:
    metadata["model_type"] = DEFAULT_MODEL_TYPE

# ——— Global State ———

FFT_Y_AXIS_MAX = 500_000

ear = None  # StreamAnalyzer instance or None

device_list = []  # Available input device names
device_indices_by_name = {}  # {device_name: system audio device index}

collected_data = {}  # { class_name: [data]}
record_val = None  # Currently selected class
is_recording = False  # Whether training data is currently being collected

model = None  # Loaded/trained model
infer_classes = []

smoother = PredictionSmoother()  # filters per-frame labels so stray frames don't count
trial_active = False  # Whether a bakeoff trial is being recorded (space bar toggles)
trial_results = set()  # Smoothed labels seen during the current trial
trial_tally = {kind: [0, 0] for kind in TRIAL_KINDS}  # {trial kind: [passed, total]}

# ——— Initialization ———
# Populate device_list with input-capable devices
audio = pyaudio.PyAudio()
for idx in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(idx)
    if info.get("maxInputChannels", 0) > 0:
        device_list.append(info["name"])
        device_indices_by_name[info["name"]] = idx

# Determine default device
default_device = device_list[0] if device_list else None
laptop_mic = next((name for name in device_list if "MacBook" in name and "Microphone" in name), None)
if laptop_mic:
    default_device = laptop_mic
saved = metadata["default_device"]
if saved in device_list:
    default_device = saved

# Instantiate audio reader ('ear') if possible
if default_device:
    device_idx = device_indices_by_name[default_device]
    ear = get_ear(device=device_idx)
else:
    ear = None

# data management


def set_infer_classes(new_classes):
    global infer_classes
    infer_classes = new_classes
    dpg.configure_item("infer_classes_list", items=infer_classes)


def format_collected_data(collected_data):
    # turns collected data into arrays for training
    x = []
    y = []
    class_names = []

    for i, (key, val) in enumerate(collected_data.items()):
        x.extend(val)
        y.extend([i for _ in range(len(val))])
        class_names.append(key)

    # maybe need to shuffle later

    x = np.array(x)
    y = np.array(y)

    return x, y, class_names


def update_recording_ui():
    """Keep the recording button and status text in sync with state."""
    if not dpg.does_item_exist("record_button"):
        return

    if is_recording:
        dpg.configure_item("record_button", label="Stop Recording")
        dpg.set_value("recording_status", f"Collecting data: {record_val}")
        dpg.configure_item("recording_status", color=(80, 180, 100))
    elif record_val:
        dpg.configure_item("record_button", label="Record Data")
        dpg.set_value("recording_status", f"Ready: {record_val}")
        dpg.configure_item("recording_status", color=(180, 180, 180))
    else:
        dpg.configure_item("record_button", label="Record Data")
        dpg.set_value("recording_status", "Select a class first")
        dpg.configure_item("recording_status", color=(180, 180, 180))


def set_recording(recording):
    global is_recording
    is_recording = bool(recording and record_val)
    update_recording_ui()


# ——— Callback Definitions ———
def device_select_callback(s, a, u):
    """Handle user selecting a new audio device."""
    device_name = a
    metadata["default_device"] = device_name
    save_metadata()
    global ear
    ear = get_ear(device=device_indices_by_name[device_name])


def model_select_callback(s, a, u):
    metadata["model_type"] = a
    save_metadata()


# Class management callbacks


def add_class_callback():
    cls = dpg.get_value("class_input").strip()
    add_class(cls)


def add_class(cls):
    if not cls or cls in collected_data:
        dpg.set_value("class_input", "")
        return
    collected_data[cls] = []
    # Create UI group for the new class
    with dpg.group(parent="class_list", tag=f"grp_{cls}"):
        dpg.add_text(f"{cls}: 0", bullet=True, tag=f"cls_{cls}")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Select", callback=lambda: select_class_callback(cls))
            dpg.add_button(label="Clear", callback=lambda: clear_class(cls))
            dpg.add_button(label="Delete", callback=lambda: delete_class(cls))
        dpg.add_separator()
    dpg.set_value("class_input", "")

    # updates the metadata
    metadata["data_classes"] = list(collected_data.keys())
    save_metadata()


def select_class_callback(cls):
    """Switch recording target to the chosen class."""
    global record_val
    record_val = cls
    dpg.set_value("selected_class", f"Selected: {cls}")
    dpg.configure_item("record_button", enabled=True)
    set_recording(False)


def record_button_callback():
    if not record_val:
        set_recording(False)
        return
    set_recording(not is_recording)


def clear_class(cls):
    collected_data[cls] = []
    dpg.set_value(f"cls_{cls}", f"{cls}: 0")


def clear_all_classes():
    for k in collected_data.keys():
        clear_class(k)


def delete_class(cls):
    global record_val
    collected_data.pop(cls, None)
    dpg.delete_item(f"grp_{cls}")

    if cls == record_val:
        record_val = None
        dpg.set_value("selected_class", f"Selected: {record_val}")
        dpg.configure_item("record_button", enabled=False)
        set_recording(False)

    # updates the metadata
    metadata["data_classes"] = list(collected_data.keys())
    save_metadata()


def delete_all_classes():
    for k in list(collected_data.keys()):
        delete_class(k)


# Mode switching
def switch_mode(selected):
    """Toggle between Train and Infer UI groups."""
    train_visible = selected == "Train"
    dpg.configure_item("train_group", show=train_visible)
    dpg.configure_item("infer_group", show=not train_visible)
    dpg.set_value("mode", selected)


# Training/inference buttons
def train_model_callback():
    # makes sure theres some data to even train with (2 class min)
    data_collected = False
    n_valid = 0
    for key, val in collected_data.items():
        if len(val) > 0:
            n_valid += 1
        if n_valid >= 2:
            data_collected = True
            break

    if not data_collected:
        dpg.set_value("training_status_text", "Need data in at least 2 classes")
        dpg.configure_item("training_status_text", show=True)
        return

    dpg.set_value("training_status_text", "Training...")
    dpg.configure_item("train_button", enabled=False)
    dpg.configure_item("training_status_text", show=True)

    def _background_training():
        X, y, class_names = format_collected_data(collected_data)
        X = preprocess_data(X)
        model_type = dpg.get_value("model_combo")

        # train and set the global model
        global model
        model, infer_classes = train_model(X, y, class_names, model_type)
        set_infer_classes(infer_classes)

        dpg.configure_item("train_button", enabled=True)
        dpg.configure_item("training_status_text", show=False)
        switch_mode("Infer")

    threading.Thread(target=_background_training, daemon=True).start()


def load_model_callback():
    dpg.configure_item("load_model_dialog", show=True)


def save_model_callback():
    dpg.configure_item("save_model_dialog", show=True)


# File dialog callbacks
def load_model_dialog_callback(s, a, u):
    path = a.get("file_path_name", None)
    print(path)

    def _background_loading():
        try:
            p = Path(path)
            global model
            model, infer_classes = load_model(p)
            set_infer_classes(infer_classes)
        except:
            print("Loading Failed. Continuing without loading any model.")

    if path:
        threading.Thread(target=_background_loading, daemon=True).start()


def save_model_dialog_callback(s, a, u):
    path = a.get("file_path_name", None)

    def _background_saving():
        p = Path(path)
        global model
        global infer_classes
        save_model(model, infer_classes, p)

    if path:
        threading.Thread(target=_background_saving, daemon=True).start()


# Bakeoff trial: space bar starts recording smoothed labels, space again stops
def update_tally_text():
    lines = [f"{kind}: {p}/{n}" for kind, (p, n) in trial_tally.items()]
    dpg.set_value("trial_tally", "\n".join(lines))


def toggle_trial():
    """Space bar handler: start a trial, or stop it and report the result set."""
    global trial_active
    if dpg.get_value("mode") != "Infer" or model is None:
        dpg.set_value("trial_status", "Train or load a model in Infer mode first")
        return
    # Don't steal the space bar while a class name is being typed
    if dpg.is_item_active("class_input") or dpg.is_item_focused("class_input"):
        return

    if not trial_active:
        trial_results.clear()
        smoother.reset()
        trial_active = True
        dpg.set_value("trial_status", "RECORDING... (space to stop)")
        dpg.configure_item("trial_status", color=(80, 180, 100))
        dpg.set_value("trial_result", "")
        return

    trial_active = False
    result_text = format_results(trial_results)
    dpg.configure_item("trial_status", color=(180, 180, 180))
    dpg.set_value("trial_status", "Stopped (space to start the next trial)")
    dpg.set_value("trial_result", f"Results: {result_text}")
    print(f"Trial results: {result_text}")


def score_trial_callback(s=None, a=None, u=None):
    """Optional self-test: score the last trial against the selected trial kind."""
    if trial_active or not trial_results:
        return
    kind = dpg.get_value("trial_kind")
    passed = evaluate_trial(kind, trial_results)
    trial_tally[kind][0] += int(passed)
    trial_tally[kind][1] += 1
    trial_results.clear()
    dpg.set_value("trial_result", f"{kind}: {'PASS' if passed else 'FAIL'}")
    update_tally_text()


def reset_tally_callback(s=None, a=None, u=None):
    for counts in trial_tally.values():
        counts[0] = counts[1] = 0
    update_tally_text()


def frame_callback():
    # get the live data
    freqs, amps = get_audio_features(ear)

    # Plot FFT amplitude against its corresponding frequency in Hz.
    dpg.set_value("fft_series", [list(freqs), list(amps)])
    if len(freqs) and len(amps):
        peak_index = int(np.argmax(amps))
        peak_frequency = float(freqs[peak_index])
        peak_amplitude = float(amps[peak_index])
        dpg.set_value(
            "fft_status",
            f"Peak: {peak_frequency:.1f} Hz / amplitude {peak_amplitude:.3g}",
        )
        dpg.set_axis_limits("x_axis", 0, float(freqs[-1]))
        dpg.set_axis_limits("y_axis", 0, FFT_Y_AXIS_MAX)

    # recording data to collected data
    if is_recording and record_val:
        # TODO: Maybe save the frequency X values for preprocessing?
        collected_data[record_val].append(amps)
        dpg.set_value(
            f"cls_{record_val}", f"{record_val}: {len(collected_data[record_val])}"
        )

    # live prediction
    if (dpg.get_value("mode") == "Infer") and model is not None:
        amps = np.array([amps])
        X = preprocess_data(amps)
        y_pred = predict(model, X)[0]
        if y_pred < len(infer_classes):
            y_label = smoother.update(infer_classes[y_pred])  # get the name
            dpg.set_value("pred_text", y_label)
            dpg.configure_item("infer_classes_list", default_value=y_label)
            if trial_active:
                trial_results.add(y_label)
        else:
            raise ValueError("Y label predicted is not in the infer classes")

    dpg.set_frame_callback(dpg.get_frame_count() + 1, frame_callback)


# Resize handler
def on_vp_resize(s, a):
    W, H = dpg.get_viewport_width(), dpg.get_viewport_height()
    w2 = int(W * .95 * 0.7)
    h2 = .95 * H
    # Position and size windows dynamically
    dpg.configure_item("fft_vis", pos=(0, 0), width=w2, height=h2)
    dpg.configure_item("ctrl_win", pos=(w2, 0), width=W - w2, height=h2)
    if dpg.does_item_exist("recording_status"):
        dpg.configure_item("recording_status", wrap=max(140, W - w2 - 24))


# ——— UI Construction ———

dpg.create_context()
dpg.create_viewport(title="FFT Trainer", width=800, height=600)

# Menu bar for device selection
with dpg.viewport_menu_bar():
    with dpg.menu(label="Options"):
        dpg.add_combo(
            items=device_list,
            label="Device",
            tag="device_combo",
            default_value=default_device,
            callback=device_select_callback,
            width=300,
        )  # TODO: ADD A CALLBACK

# FFT Visualization window
with dpg.window(
    tag="fft_vis", label="FFT Spectrum", no_move=True, no_resize=True, no_collapse=True
):
    dpg.add_text("Real-time FFT Signal", bullet=True)
    dpg.add_text("Waiting for microphone data...", tag="fft_status")
    with dpg.plot(label="FFT Plot", height=-1, width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis", label="Frequency (Hz)")
        dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis", label="Amplitude")
        dpg.set_axis_limits("y_axis", 0, FFT_Y_AXIS_MAX)
        dpg.add_line_series([], [], label="FFT", parent="y_axis", tag="fft_series")

# Control window with Train/Infer groups
with dpg.window(
    tag="ctrl_win", label="Controls", no_move=True, no_resize=True, no_collapse=True
):
    # Mode switch combo
    dpg.add_text("Mode:")
    dpg.add_combo(
        items=["Train", "Infer"],
        default_value="Train",
        tag="mode",
        callback=lambda s, a, u: switch_mode(a),
    )
    dpg.add_separator()

    # Training UI group
    with dpg.group(tag="train_group"):
        dpg.add_text("Model:")
        dpg.add_combo(
            items=MODEL_OPTIONS,
            default_value=metadata["model_type"],
            tag="model_combo",
            callback=model_select_callback,
            width=-1,
        )
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="class_input")
            dpg.add_button(label="Add Class", callback=add_class_callback)
        dpg.add_separator()
        dpg.add_text("Classes & Data:", bullet=True)
        dpg.add_child_window(tag="class_list", width=-1, height=200)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Clear All", callback=clear_all_classes)
            dpg.add_button(label="Delete All", callback=delete_all_classes)
        dpg.add_separator()
        dpg.add_text(f"Selected: {record_val}", tag="selected_class")
        dpg.add_button(
            label="Record Data",
            tag="record_button",
            enabled=False,
            callback=record_button_callback,
        )
        dpg.add_text(
            "Select a class first",
            tag="recording_status",
            color=(180, 180, 180),
            wrap=220,
        )
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Train Model", tag="train_button", callback=train_model_callback
            )
            dpg.add_text("Training...", tag="training_status_text", show=False)

    # Inference UI group
    with dpg.group(tag="infer_group", show=False):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Save Model", callback=save_model_callback, tag="btn_save"
            )
            dpg.add_button(
                label="Load Model", callback=load_model_callback, tag="btn_load"
            )
        dpg.add_text("Classes:", bullet=True)
        dpg.add_listbox(
            items=infer_classes,
            tag="infer_classes_list",
            num_items=5,
        )
        with dpg.group(horizontal=True):
            dpg.add_text("Prediction: ")
            dpg.add_text("---", tag="pred_text")
        dpg.add_separator()
        dpg.add_text("Bakeoff trial (space bar):", bullet=True)
        dpg.add_text(
            "Train or load a model first",
            tag="trial_status",
            color=(180, 180, 180),
            wrap=220,
        )
        dpg.add_text("", tag="trial_result", wrap=220)
        dpg.add_combo(items=TRIAL_KINDS, default_value=TRIAL_KINDS[0], tag="trial_kind", width=-1)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Score Trial", callback=score_trial_callback)
            dpg.add_button(label="Reset Tally", callback=reset_tally_callback)
        dpg.add_text("", tag="trial_tally")

# File dialogs
with dpg.file_dialog(
    tag="load_model_dialog",
    show=False,
    callback=load_model_dialog_callback,
    width=700,
    height=400,
    modal=True,
    default_path=".",
    label="Choose Model to Load",
):
    dpg.add_file_extension(".model")

with dpg.file_dialog(
    tag="save_model_dialog",
    show=False,
    callback=save_model_dialog_callback,
    label="Choose Save Location",
    width=700,
    height=400,
    modal=True,
    default_filename="model",
    file_count=1,
    default_path=".",
):
    dpg.add_file_extension(".model")

# Setup callbacks and start
with dpg.handler_registry():
    dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=lambda: toggle_trial())
update_tally_text()
dpg.set_viewport_resize_callback(on_vp_resize)
dpg.set_frame_callback(1, frame_callback)
on_vp_resize(None, None)  # initial layout

# loading initial dataclasses
for c in metadata["data_classes"]:
    add_class(c)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
