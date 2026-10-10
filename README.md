# University IT Policy Q&A

This program answers questions about the policy file supplied with SCSE GEX3. It retrieves three passages, drafts an answer with Qwen3-0.6B, and asks the model to review the answer against those passages. A failed review allows one revision. If that still fails, the program reports the failure instead of presenting the draft as an answer.

The model uses its thinking mode. A factual answer cites one retrieved policy ID and is reviewed against that cited passage. For an answer that says it does not know, a separate review first asks whether the retrieved excerpts contain the requested fact. If they do, the reviewer must select an unchanged passage as evidence and the answer needs revision. Missing information permits the unknown answer. The recorded decision distinguishes this rule from the model's raw response.

The file contains example policies for the exercise, not verified guidance from a real university. It does not give a Service Desk phone number.

## Run it locally

Use Python 3.11 or newer and Ollama. In the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:0.6b
ollama pull nomic-embed-text
.\.venv\Scripts\python.exe check_data.py
.\.venv\Scripts\python.exe rag.py
.\.venv\Scripts\python.exe agent.py "Which wireless network should guests use?"
```

Ollama must be running before indexing or asking questions. The default address is `http://127.0.0.1:11434`; set `OLLAMA_BASE_URL` to use another local address. Model calls request GPU offloading, but availability depends on the local Ollama installation and driver.

The first indexing run processes 1,196 policy blocks. Later runs reuse that index, or resume a partial build. The program checks saved text and metadata before using the index. A changed policy file gets a separate collection. Do not add other documents to this collection.

Answers include policy IDs. Each question also writes a JSON record under `runs/`, including retrieved passages, final model responses, reviews and any revision. A missing fact should produce `I don't know.`; a connection error or failed review is reported as an error, not disguised as a successful unknown answer.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_live.py --output runs/check.json
```

Unit tests use fake model responses and need no model service. The live check makes real requests for a phone number, print retention, guest Wi-Fi and an account lock. It then gives the reviewer a wrong network name and an unnecessary unknown answer, both of which should be rejected. It also checks that both models report full GPU offloading. Use a new output filename on each run; previous results are not overwritten.

The local numeric and citation checks catch some unsupported details. They cannot prove that every sentence is correct, and the model-based review can also make mistakes.

A recorded run is included in [validation](validation/README.md), with the original final responses and review decisions.

## Guarded Windows runs

`run_local.ps1` needs PowerShell 7, an NVIDIA GPU and locally downloaded Ollama models. It starts its own loopback-only service, samples available memory and temperature, and closes the service it started when the run ends or fails. It refuses an occupied port and does not stop existing services.

```powershell
pwsh -File .\run_local.ps1 -Python .\.venv\Scripts\python.exe -Task index
pwsh -File .\run_local.ps1 -Python .\.venv\Scripts\python.exe -Task verify
```

The script requires 6 GiB free RAM, 5 GiB free VRAM and a starting GPU temperature no higher than 68 C. It stops its worker if free RAM falls below 3 GiB, free VRAM below 256 MiB, or temperature reaches 82 C. These are sampled safeguards, not a guarantee against every hardware problem.

On the development laptop, the installed NVIDIA driver could not run Ollama's CUDA backend. Vulkan worked without a driver change:

```powershell
pwsh -File .\run_local.ps1 -Python .\.venv\Scripts\python.exe -Task index -Backend vulkan -VulkanDevice 1
pwsh -File .\run_local.ps1 -Python .\.venv\Scripts\python.exe -Task verify -Backend vulkan -VulkanDevice 1
```

The Vulkan device number is machine-specific. On this laptop, device 1 is the RTX 3060 and device 0 is the integrated GPU. These settings apply only to the temporary processes.

## Files

- `policy_loader.py` and the two supplied data files are retained from the course repository.
- `rag.py` manages the Chroma index and similarity search.
- `rag_tool.py` provides the policy search tool.
- `agent.py` runs generation, review and the optional revision.
- `source_manifest.json` records the original source revision and file checksums.

The reference JSONL file is used by the data checks only. Retrieval uses `university_it_policies.txt` through the supplied loader. Index files, model files, environments and local run logs are not included in Git.
