---
title: Tab Agent - AI Tablature Transcription (MVP)
emoji: 🎸
colorFrom: blue
colorTo: purple
sdk: docker
app_file: DOCKERFILE.dockerfile
license: mit
---

# 🎸 Tab Agent - AI Tablature Transcription (MVP)

**Production-ready AI-powered audio-to-tablature transcription for guitar and bass using Basic Pitch.**

Upload an audio file and get accurate, playable tablature with technique detection and optimal fingering!

## ⏱️ Processing Times

Audio transcription runs on CPU (free tier). Expect:
- **30-second clip:** ~15 seconds
- **3-minute song:** ~90 seconds
- **Full-length track:** 2-4 minutes

Processing is slower than GPU-accelerated services but completely free. Grab a coffee while it works. ☕

## 🚀 Try It Now

**Live on HuggingFace Spaces:** [scottymills-tab-agent-pro.hf.space](https://scottymills-tab-agent-pro.hf.space)

Upload an audio file, select Guitar or Bass, and get tablature in seconds.

## ✨ Features

- 🤖 **Basic Pitch AI Model** - Spotify's proven production-ready transcription model
- ⚡ **Zero GPU Acceleration** - Faster processing with Hugging Face Zero GPU
- 🎵 **Multi-Stage Pipeline** - Demucs stem separation → spatial audio processing → AI transcription → dynamic programming
- 🎸 **Multi-Track Support** - Separate transcriptions for lead guitar, rhythm guitars (L/R), and bass
- 📝 **Multiple Export Formats** - MIDI, ASCII tablature, JSON
- 🎯 **Optimal Fingering** - Viterbi algorithm finds most playable fingering patterns
- ✨ **Technique Detection** - Automatically detects slides, hammer-ons, pull-offs
- 🎯 **Suno-Aware** - Enhanced processing for AI-generated audio (Suno, Udio)

## 🚀 How to Use

1. **Upload** an audio file (WAV, MP3, FLAC, etc.)
2. **Select** instrument type (Guitar or Bass)
3. **Choose** export formats (MIDI, Tab, JSON)
4. **Click** "Transcribe to Tablature"
5. **Download** the ZIP file with all outputs!

## 📊 Processing Times

| Duration | Estimated Time (Zero GPU) |
|----------|---------------------------|
| 30 sec   | ~1-2 minutes              |
| 60 sec   | ~2-3 minutes              |
| 3 min    | ~6-8 minutes              |

**Powered by:** Zero GPU for faster processing on HF Spaces

## 🎯 Best Results Tips

- ✅ Use high-quality audio (WAV/FLAC preferred)
- ✅ Clean recordings work better than live/noisy audio
- ✅ Isolated guitar/bass tracks give best accuracy
- ✅ Shorter clips (< 60 seconds) process faster
- ✅ **Suno AI audio supported** - automatic artifact detection and cleanup
- ❌ Avoid heavily distorted or compressed audio

## 🏗️ How It Works

### Processing Pipeline

1. **Quality Analysis** - Detects AI-generated artifacts (Suno, Udio) and applies preprocessing
2. **Stem Separation (Demucs)** - Isolates guitar/bass from full mix
3. **Spatial Processing** - Separates lead and rhythm guitars using mid-side technique
4. **AI Transcription (Basic Pitch)** - Converts audio to MIDI notes with proven accuracy
5. **Post-Processing** - Removes octave errors and spurious notes common in AI audio
6. **Tablature Generation** - Dynamic programming finds optimal fingering
7. **Technique Detection** - Identifies slides, hammer-ons, pull-offs

### Why Basic Pitch for MVP?

| Feature | Basic Pitch | Notes |
|---------|-------------|-------|
| Reliability | ✅ Production-ready | Used by Spotify, proven at scale |
| Dependencies | ✅ Lightweight | TensorFlow-based, easy deployment |
| Guitar Accuracy | ✅ Good (70-85%) | Solid baseline for clean audio |
| Suno Support | ✅ Enhanced | Custom post-processing for AI audio |

## 📦 Install Locally

Want faster processing or GPU acceleration? Install Tab Agent locally:

```bash
git clone https://github.com/YOUR_USERNAME/Tab-Agent.git
cd Tab-Agent
pip install -r requirements_working.txt
python main.py input/your_song.wav
```

### Reaper Integration (ReaPack)

Install directly in Reaper DAW:

1. Extensions → ReaPack → Import repositories
2. Add URL: `https://raw.githubusercontent.com/YOUR_USERNAME/Tab-Agent/main/index.xml`
3. Browse packages → Install "Tab Agent"
4. Select audio → Run script!

## 🔗 Links

- **GitHub**: [Tab-Agent Repository](https://github.com/YOUR_USERNAME/Tab-Agent)
- **Documentation**: [Full Guide](https://github.com/YOUR_USERNAME/Tab-Agent/blob/main/README.md)
- **Basic Pitch**: [Spotify Research](https://github.com/spotify/basic-pitch)

## 📄 License

MIT License - Free for personal and commercial use

## 🙏 Acknowledgments

- **Spotify Research** - For Basic Pitch transcription model
- **Meta AI** - For Demucs source separation
- **Hugging Face** - For Zero GPU hosting and infrastructure

---

<div align="center">

**Made with ❤️ and 🤖 AI**

[⭐ Star on GitHub](https://github.com/YOUR_USERNAME/Tab-Agent) • [📚 Documentation](https://github.com/YOUR_USERNAME/Tab-Agent/blob/main/README.md) • [📦 Reaper Plugin](https://github.com/YOUR_USERNAME/Tab-Agent#reaper-integration)

</div>
