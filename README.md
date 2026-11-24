# Inkling 2 Mini on a Mac: 37 → 112 tok/sec

![Inkling 2 Mini on a Mac, decode speed commit by commit](media/inkling2-mac.gif)

This branch (`metal-inkling2-fuse`) is Unsloth's llama.cpp release `b12588-mix-c0c3de8` with the Metal backend reworked
for [Inkling 2 Mini](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). On an M5 Ultra, one stream of `UD-Q4_K_XL`
decodes at 112 tok/sec, where the release it started from does 37. The model's output is unchanged within rounding
(KL divergence 0.0009 against the untouched build, the same as that build against itself with fusion off).

Where the speed came from: the model's small ops are fused into single Metal kernels (7,177 → 1,466 kernels a token),
the graph is ordered by depth so independent branches run side by side, memory barriers are only placed where ranges
really overlap, and the command buffer for the next token is encoded while the GPU runs the current one.

### Build

```bash
git clone -b metal-inkling2-fuse https://github.com/jhughes2112/llama.cpp-unsloth
cd llama.cpp-unsloth
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Needs macOS on Apple Silicon, Xcode command line tools and CMake. Do not clone under `/tmp`: the web UI packaging step
of the server build fails there.

### Get the model

```bash
pip install -U huggingface_hub
hf download unsloth/Inkling-2-Mini-GGUF Old_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file     Old_Support/mmproj-BF16.gguf_file --local-dir models
mv models/Old_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file models/Inkling-2-Mini-UD-Q4_K_XL.gguf
mv models/Old_Support/mmproj-BF16.gguf_file models/mmproj-BF16.gguf
```

17.6 GB of weights plus the 0.6 GB image projector. This release reads the GGUF layout Unsloth first published, which
now lives in the `Old_Support` folder of their repo (the root files have a newer layout and will not load here).

### Run

```bash
# measure decode speed
build/bin/llama-bench -m models/Inkling-2-Mini-UD-Q4_K_XL.gguf -p 0 -n 400 -r 3

# chat in the terminal
build/bin/llama-cli -m models/Inkling-2-Mini-UD-Q4_K_XL.gguf -ngl 99 -fa on --jinja

# OpenAI-compatible server with image input, at http://127.0.0.1:8080
GGML_METAL_N_CB=6 build/bin/llama-server -m models/Inkling-2-Mini-UD-Q4_K_XL.gguf \
    --mmproj models/mmproj-BF16.gguf -c 32768 -ngl 99 -fa on --jinja
```

### Good to know

- 112 tok/sec is an M5 Ultra at an empty context; it is 106 at 4K tokens of context and 93 at 32K. Smaller Apple
  chips will be slower. Prompt reading is about 1,900 tok/sec.
- Only the Metal backend changed. CUDA, Vulkan and CPU run as in the Unsloth release.
- Leave the K/V cache at its default (F16). `-ctk q8_0 -ctv q8_0` halves cache memory at long contexts but needs
  `LLAMA_ATTN_ROT_DISABLE=1` to stay on the fast path and costs a little quality.
- `GGML_METAL_PROFILE=1` prints GPU time, kernels and serial steps a token. `GGML_METAL_FUSION_DISABLE=1` runs
  without the fused kernels, for comparison.
- The base is a frozen release, so this branch does not follow upstream llama.cpp.
- `media/make_inkling2_gif.py` draws the chart above from the numbers in the commit messages.

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
