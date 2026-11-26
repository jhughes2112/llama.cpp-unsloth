# Inkling 2 Mini on a Mac: 37 → 137 tok/sec on one stream (M5 Ultra)

![Inkling 2 Mini on a Mac, decode speed commit by commit](media/inkling2-mac.gif)

This branch is Unsloth's llama.cpp release `b12588-mix-c0c3de8` with the Metal backend reworked for
[Inkling 2 Mini](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). **All numbers on this page are one stream on an
Apple M5 Ultra**, `llama-server`, at the model's own sampling settings (temperature 1.0; nothing was measured greedy).
Smaller Apple chips will be slower.

The model's output is unchanged within rounding: KL divergence 0.0008 against the untouched build at batch 1, and 0.0019
at the default batch (the untouched build against itself with fusion off: 0.0008 and 0.0016).

### Speed, one stream, M5 Ultra

Decode tok/sec by how much context comes before the reply (a real C++ or Markdown context of that many tokens, then a
code or prose task, 300 tokens generated, mean of three seeds):

| context | code | prose |
|---|---|---|
| short prompt | 137 | 136 |
| 1K tokens | 128 | 128 |
| 4K tokens | 126 | 127 |
| 8K tokens | 124 | 124 |
| 32K tokens | 110 | 110 |

The speed is the same for every kind of text; the rows are three seeds each.

The untouched Unsloth release decodes 37 tok/sec on the same machine.

### What you need

- macOS on Apple Silicon, Xcode command line tools, CMake, Python 3 with `huggingface_hub`.
- The model: Unsloth's **`UD-Q4_K_M`** GGUF from the **`Old_Support`** folder (16.6 GB) and its image projector
  `mmproj-BF16.gguf` (0.6 GB). The files in the root of that repo have a newer layout and will not load here.

### Build

```bash
git clone -b metal-inkling2-fuse https://github.com/jhughes2112/llama.cpp-unsloth
cd llama.cpp-unsloth
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Do not clone under `/tmp`: the web UI packaging step of the server build fails there.

### Get the model (exactly these two files)

```bash
pip install -U huggingface_hub
hf download unsloth/Inkling-2-Mini-GGUF Old_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file \
    Old_Support/mmproj-BF16.gguf_file --local-dir models
mv models/Old_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file models/Inkling-2-Mini-UD-Q4_K_M.gguf
mv models/Old_Support/mmproj-BF16.gguf_file models/mmproj-BF16.gguf
```

### Run

```bash
# OpenAI-compatible server with image input, at http://127.0.0.1:8080
build/bin/llama-server -m models/Inkling-2-Mini-UD-Q4_K_M.gguf --mmproj models/mmproj-BF16.gguf \
    -c 32768 -ngl 99 -fa on --jinja -t 2 -ot "token_embd\.weight=MTL0" --backend-sampling

# measure decode speed
build/bin/llama-bench -m models/Inkling-2-Mini-UD-Q4_K_M.gguf -p 0 -n 400 -r 3
```

- `--backend-sampling` draws the next token on the GPU, so nothing of the 200K-token vocabulary goes back to the CPU.
- `-ot "token_embd\.weight=MTL0"` keeps the token embeddings on the GPU, `-t 2` is enough CPU threads for what is left.

### Credits

- [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for Inkling 2 Mini.
- [Unsloth](https://github.com/unslothai/llama.cpp) for the llama.cpp release this branch is built on, its Inkling 2
  support, and the `UD-Q4_K_M` quant.
- The llama.cpp and ggml authors.

### Good to know

- Where the speed came from: the model's small ops are fused into single Metal kernels (541 dependent GPU steps a token,
  from 920), per-channel work runs in the epilogues of the mat-vecs, the token is drawn on the GPU, the CPU waits by
  polling, and the command buffer for the next token is encoded while the GPU runs the current one.
- Only the Metal backend changed. CUDA, Vulkan and CPU run as in the Unsloth release.
- Leave the K/V cache at its default (F16).
- `GGML_METAL_PROFILE=1` prints GPU time, kernels and serial steps a token. `GGML_METAL_FUSION_DISABLE=1` runs
  without the fused kernels, for comparison.
- The base is a frozen release, so this branch does not follow upstream llama.cpp.
- `media/make_inkling2_gif.py` draws the chart above (its history ends before the latest work).

---

# llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
