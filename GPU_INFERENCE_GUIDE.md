# GPU Inference Guide for Jarvis AI

This guide explains how to enable GPU-accelerated AI inference using Ollama on your system.

## Important: GPU Setup is Server-Side Only

**Ollama's GPU acceleration is configured when starting the Ollama server, NOT through API request parameters.** The `pi_app.py` chat request payload does not need (and does not support) explicit GPU device hints. Ollama automatically detects and uses available GPU hardware if the server is started correctly.

## Prerequisites

### Windows GPU Setup

#### NVIDIA GPUs (Recommended)
1. **Install NVIDIA GPU drivers** (531+, or 570+ for older cards)
   - Download from: https://www.nvidia.com/Download/driverDetails.aspx
   - Verify: Run `nvidia-smi` in PowerShell
   
2. **Install Ollama**
   - Download from: https://ollama.ai
   - Ollama automatically includes NVIDIA CUDA support

3. **Start Ollama with GPU acceleration** (automatic if installed correctly)
   ```powershell
   ollama serve
   ```
   - Check the log output for: `"GPU": "NVIDIA"` or similar
   - If you see `"GPU": "CPU"`, the GPU is not being used

#### AMD GPUs (Experimental)
1. **Install AMD GPU drivers**
2. **Install Ollama** (ROCM support available on Linux)
3. Start Ollama: `ollama serve`

#### No GPU (CPU-Only)
- Ollama will run on CPU by default
- AI responses will be slower but still functional
- Use smaller models like `phi3:mini` for acceptable performance

### Linux (Raspberry Pi) GPU Setup

#### NVIDIA Jetson Boards
1. **System should come with NVIDIA drivers pre-installed**
2. **Install Ollama ARM version**: https://ollama.ai/download/linux
3. Start: `ollama serve`
4. Check GPU status: `nvidia-smi`

#### Raspberry Pi 4/5 (BCM GPU)
- Native GPU inference is **not currently supported** by Ollama
- Use CPU inference with smaller models
- Consider overclocking for better performance

#### Other Linux Systems
- Follow NVIDIA or AMD setup based on your GPU
- Verify drivers: `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)

## Starting Ollama with GPU Support

### Windows PowerShell

```powershell
# Start Ollama with automatic GPU detection
ollama serve

# Output should show something like:
# {"level":"info","ts":1234567890.123,"msg":"GPU":"NVIDIA"}
```

### Linux / Raspberry Pi

```bash
# Start Ollama
ollama serve

# Check GPU status
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD
```

## Monitoring GPU Inference

### In pi_app.py

**GPU temperature is already displayed on-screen** in the top-right stats overlay:
- `GPU: --°C` = GPU not available or not being used
- `GPU: 45.0°C` = GPU is active and being used for inference

The temperature indicator automatically:
- Changes color based on temperature (green → yellow → red)
- Detects GPU via `nvidia-smi` (Windows/Linux) or `/sys/class/thermal/` (Raspberry Pi)
- Displays N/A if GPU is not available

### Command-Line GPU Monitoring

```bash
# Windows: Check GPU usage while running
nvidia-smi -l 1  # Update every 1 second

# Linux
watch -n 1 nvidia-smi
```

## Running Jarvis with GPU-Backed AI

### Step 1: Start Ollama Server (GPU enabled)
```powershell
# Windows
ollama serve

# Linux
ollama serve
```

### Step 2: Start Jarvis in another terminal
```powershell
# Use your preferred model
python pi_app.py --model phi3:mini

# Or with explicit settings
python pi_app.py --host 0.0.0.0 --port 8765 --model phi3:mini
```

### Step 3: Monitor GPU Status
- **Watch the top-right stats**: GPU temperature shows real-time GPU usage
- **CLI monitoring** (in third terminal):
  ```powershell
  nvidia-smi -l 1  # GPU stats every 1 second
  ```

## Troubleshooting

### GPU Not Detected

**Problem**: GPU shows as `--°C` in stats overlay

**Solutions**:
1. Verify Ollama is running with GPU support:
   ```bash
   ollama serve
   # Check log for "GPU":"NVIDIA" or similar
   ```

2. Check driver installation:
   ```bash
   nvidia-smi  # Should show GPU info
   ```

3. If nvidia-smi shows GPU but Ollama doesn't use it:
   - Restart Ollama: `ollama serve`
   - Check Ollama log output for driver compatibility issues

### GPU Shows But Performance is Slow

**Possible causes**:
1. Model too large for GPU memory → Use smaller model
   - Try: `ollama run phi3:mini`
   
2. GPU is underutilized → GPU is working but model is CPU-bound
   - This is normal for small models like phi3:mini

3. Network bottleneck → Check network speed between client and server

### "CUDA out of memory" errors

**Solutions**:
1. Use a smaller model: `ollama run phi3:mini`
2. Reduce context length in `ask_ollama()` options
3. Free up GPU memory: Close other GPU-using applications

## Performance Expectations

### GPU vs CPU Performance

| Model | GPU (NVIDIA) | CPU |
|-------|-------------|-----|
| phi3:mini | 1-2 sec | 3-5 sec |
| llama2 | 3-5 sec | 15-30 sec |
| mistral | 2-4 sec | 10-20 sec |

### GPU Memory Requirements

| Model | VRAM Needed |
|-------|-------------|
| phi3:mini | 2-3 GB |
| llama2 | 4-6 GB |
| mistral | 5-7 GB |

## Advanced: Per-Model GPU Memory Optimization

Ollama CLI offers memory control:

```bash
# Set max GPU memory allocation
CUDA_VISIBLE_DEVICES=0 ollama serve  # Use GPU 0 only

# For multi-GPU systems
CUDA_VISIBLE_DEVICES=0,1 ollama serve  # Use GPUs 0 and 1
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  pi_app.py (Jarvis Server)                         │
│  - Sends chat requests to Ollama API               │
│  - Reads GPU temperature (nvidia-smi, etc.)        │
│  - Displays GPU stats in on-screen overlay         │
└────────────┬────────────────────────────────────────┘
             │ HTTP POST /api/chat
             ↓
┌─────────────────────────────────────────────────────┐
│  Ollama Server (127.0.0.1:11434)                   │
│  - Detects GPU at startup (nvidia-smi checks)      │
│  - Loads model into VRAM if GPU available          │
│  - Runs inference on GPU if possible               │
│  - Falls back to CPU if needed                     │
└─────────────────────────────────────────────────────┘
             │
             ├─→ GPU Inference (if NVIDIA/AMD GPU available)
             └─→ CPU Inference (fallback or default)
```

## Key Points Summary

✅ **GPU support is automatic** - Just run `ollama serve` on a system with GPU drivers  
✅ **No code changes needed** - The `pi_app.py` request payload is correct  
✅ **Temperature monitoring works** - GPU stats display on-screen automatically  
✅ **Fallback is automatic** - If GPU fails, Ollama uses CPU seamlessly  

❌ **Cannot force GPU in request payload** - Ollama API doesn't support per-request device hints  
❌ **Don't modify `ask_ollama()` options for GPU** - This is server-side only  

## Next Steps

1. **Install NVIDIA drivers** if you have a GPU
2. **Download Ollama** from https://ollama.ai
3. **Start Ollama**: `ollama serve`
4. **Run Jarvis**: `python pi_app.py`
5. **Watch GPU stats** in the top-right overlay

---

**Last Updated**: 2024
**Ollama Version Tested**: 0.21.0+
