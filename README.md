# Kage Bunshin no Jutsu - Shadow Clone Video App

A real-time video motion detection application that recognizes the **Kage Bunshin** (Shadow Clone) hand sign from Naruto. When you perform the jutsu gesture, your webcam feed clones into **5 shadow clones** on screen!

## Hand Sign

The canonical Kage Bunshin hand sign:

- **Both hands**: Index and middle fingers extended in a **V-formation**
- **Both hands**: Ring and pinky fingers curled/folded
- **Hands interlock**: Position hands so they cross at the palms, forming an **X pattern**

## Setup

```bash
cd kagebunsih
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

- Use your webcam in a well-lit environment
- Perform the hand sign and **hold it** for about 0.25 seconds
- 5 shadow clones will appear for ~3 seconds
- Press **`q`** to quit

## Tech Stack

- **OpenCV** – Video capture and display
- **MediaPipe** – Hand landmark detection
- **NumPy** – Image processing

## Files

- `main.py` – Main application with webcam loop and clone effect
- `gesture_detector.py` – Kage Bunshin hand sign recognition
- `requirements.txt` – Python dependencies

---

*多重影分身の術 (Tajū Kage Bunshin no Jutsu)*
