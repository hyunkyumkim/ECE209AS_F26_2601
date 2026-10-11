import argparse
from pathlib import Path

from models import export_model_header, load_model


def main():
    parser = argparse.ArgumentParser(
        description="Export a saved LinearSVC .model file to an ESP32 C++ header."
    )
    parser.add_argument("model_path", type=Path, help="Path to a saved .model file")
    parser.add_argument("header_path", type=Path, help="Output .h file path")
    args = parser.parse_args()

    header_path = args.header_path
    if header_path.suffix != ".h":
        header_path = header_path.with_suffix(".h")

    model, classes = load_model(args.model_path)
    export_model_header(model, classes, header_path)
    print(f"Exported {header_path}")


if __name__ == "__main__":
    main()
