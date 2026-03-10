# Wind-Vision

Predicting wind speed from webcam imagery of Lake Garda (Malcesine Nord)
using convolutional neural networks.

## Idea

Wind creates characteristic patterns on water — small ripples at low speeds,
whitecaps and foam at higher speeds.  A camera pointed at the lake surface
therefore contains enough visual information to estimate the current wind
speed.  This project automates the entire pipeline from raw image collection
to a trained regression model.

## Pipeline

```
1. fetch      Playwright scrapes historical webcam screenshots (one per day)
2. extract    EasyOCR reads the weather overlay → CSV with wind/gust labels
3. train      ResNet-18 (pretrained) is fine-tuned on the water crops
```

## Project layout

```
src/wind_vision/
├── core/              config loading, structured logging
├── data/
│   ├── fetcher.py     browser automation (Playwright)
│   └── extract_wind.py    OCR label extraction (EasyOCR + OpenCV)
├── models/
│   ├── dataset.py     PyTorch Dataset (water-crop + wind label)
│   └── train.py       training loop (ResNet-18 → regression)
└── cli.py             unified CLI entry point
```

## Quick start

```bash
make setup          # venv + deps + playwright browser
make fetch          # download ~5 years of daily screenshots
make extract        # run OCR → data/processed_wind.csv
make train          # fine-tune ResNet-18 on water crops
```

## Tech stack

| Area            | Tools                                       |
|-----------------|---------------------------------------------|
| ML framework    | PyTorch, torchvision                        |
| Image processing| OpenCV, Pillow                              |
| OCR             | EasyOCR                                     |
| Browser automation | Playwright                               |
| Config          | PyYAML                                      |
| Packaging       | pyproject.toml, setuptools                  |

## Requirements

- Python ≥ 3.10
- See `pyproject.toml` for the full dependency list.

## License

MIT
