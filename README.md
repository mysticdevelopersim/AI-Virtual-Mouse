
Experience the future of control—wave your hand and navigate your world effortlessly.

# 🖐️ AI Virtual Mouse

> **Control your computer with just a wave of your hand. No mouse? No problem.**

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00ADD8?style=for-the-badge)

**Point. Click. Command.**

</div>

---

## 🎯 What It Does

Transform your webcam into a motion-sensing controller. Your index finger becomes your cursor, and a pinch gesture triggers clicks. Welcome to the future of human-computer interaction.

### ✨ Features

- 🎮 **Gesture Control** - Navigate your screen with hand movements
- 👆 **Air Clicking** - Pinch to click, naturally
- 🎯 **Precision Tracking** - Smooth cursor movement with adjustable sensitivity
- 📊 **Real-Time FPS** - Monitor performance on-screen
- 🖼️ **Visual Feedback** - See your hand landmarks in action

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam
- Your hands (1 minimum)

### Installation

```bash
# Clone the magic
git clone https://github.com/mysticdevelopersim/AI-Virtual-Mouse.git
cd AI-Virtual-Mouse

# Install dependencies
pip install -r requirements.txt

# Run
python AI-Virtual-Mouse.py
```

---

## 🕹️ How to Use

| Gesture | Action |
|---------|--------|
| ☝️ **Index Finger Up** | Move cursor |
| ✌️ **Index + Middle Finger Up** | Hover mode |
| 🤏 **Pinch Fingers Together** | Left click |
| ✋ **Close Fist** | Idle mode |

**Pro Tip:** Press `Q` to exit like a pro.

---

## ⚙️ Configuration

Tweak these parameters in the code for your perfect setup:

```python
frameR = 100        # Detection boundary (lower = larger tracking area)
smoothening = 10    # Cursor smoothness (higher = smoother, slower)
```

## 🛠️ Tech Stack

- **OpenCV** - Computer vision powerhouse
- **MediaPipe** - Google's ML hand tracking
- **Autopy** - Cross-platform mouse control
- **NumPy** - Mathematical wizardry

---

## 🐛 Troubleshooting
  
**Webcam not detected?** 📹 Check camera permissions

---

## 🎓 Learning Resources

Built this project to understand:
- Hand landmark detection using MediaPipe
- Real-time computer vision with OpenCV
- Coordinate mapping and interpolation
- Gesture recognition algorithms

---

## 🤝 Contributing

Found a bug? Have an idea? PRs welcome!

1. Fork it
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 💡 Future Enhancements

- [ ] Custom gesture programming
- [ ] Voice commands integration

---

<div align="center">

*If you found this project helpful, give it a ⭐!*

</div>
