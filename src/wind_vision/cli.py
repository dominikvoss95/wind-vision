"""CLI entry point for the wind-vision pipeline."""

import argparse
import asyncio
import logging

from wind_vision.data.fetcher import run_fetch
from wind_vision.data.extract_wind import main as run_extraction
from wind_vision.models.train import train_model
from wind_vision.models.evaluate import evaluate
from wind_vision.models.predict import predict_wind
from wind_vision.models.explain import run_explanation


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(prog="wind-vision")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("fetch", help="download historical webcam screenshots")
    sub.add_parser("extract", help="run OCR to build the label CSV")
    sub.add_parser("train", help="train the wind speed model")
    sub.add_parser("eval", help="evaluate the trained model")
    
    predict_parser = sub.add_parser("predict", help="predict wind speed for a specific image")
    predict_parser.add_argument("image_path", help="path to the webcam image")

    explain_parser = sub.add_parser("explain", help="visualize model focus (Grad-CAM)")
    explain_parser.add_argument("image_path", help="path to image")

    args = parser.parse_args()

    if args.cmd == "fetch":
        asyncio.run(run_fetch())
    elif args.cmd == "extract":
        run_extraction()
    elif args.cmd == "train":
        train_model(
            csv_file="data/processed_wind.csv",
            img_dir="data/raw/webcam",
        )
    elif args.cmd == "eval":
        evaluate(
            csv_file="data/processed_wind.csv",
            img_dir="data/raw/webcam",
            weights="models/wind_model.pth",
        )
    elif args.cmd == "predict":
        prediction = predict_wind(args.image_path)
        print(f"\nPredicted Wind Speed: {prediction:.1f} kts\n")
    elif args.cmd == "explain":
        run_explanation(args.image_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
