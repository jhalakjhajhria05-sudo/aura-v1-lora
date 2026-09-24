---
license: apache-2.0
base_model: sarvamai/sarvam-2b
library_name: peft
tags:
- agent
- hinglish
- indian-ai
- system-agent
- qlora
- build-in-public
language:
- hi
- en
pipeline_tag: text-generation
---

<div align="center">

<img src="aura_logo.png" alt="AURA logo" width="420"/>

# AURA — v1 (LoRA adapter)

**A Hinglish system-agent that controls your PC — built from scratch on free Kaggle GPUs.**

*"bhai chrome kholo"* — and the computer obeys.

</div>

---

## What is this?

A **LoRA adapter** for [Sarvam-2B](https://huggingface.co/sarvamai/sarvam-2b) (India's open-source LLM, Apache 2.0) that turns it into **AURA** — a system agent that understands Hindi, Hinglish and English commands and executes real actions on the machine:

| Skill | What it does |
|---|---|
| 📁 Aura Manager | create / save / delete / move / organize files |
| 📦 Aura Installer | install & uninstall software (guarded) |
| 🚀 Aura Launchpad | open apps, websites, browser searches |
| 🔍 Aura Researcher | web search & page fetch |
| 🧮 Aura Aryabhatta | math via a Python sandbox |
| ⚙️ Aura Controller | screenshots, clipboard, volume, system info |

Plus personality: AURA replies in natural Hinglish, and knows when to just talk instead of calling a skill.

## Safety design

- Destructive commands (delete, install, format) require **explicit user confirmation**
- Installer downloads are restricted to an **official-domains allowlist**
- Dangerous patterns (`rm -rf /`, `format`, fork bombs...) are **hard-blocked**
- The Python sandbox has a blocklist + timeout

## Training

| | |
|---|---|
| Method | QLoRA (r=16, alpha=32) + NEFTune (alpha=5) + prompt-masked loss |
| Data | 6,687 examples — 6 skills + chitchat, 3 languages (Hindi / Hinglish / English), incl. multi-turn `[TOOL RESULT] → confirmation` patterns |
| Hardware | Free Kaggle T4 x2 — total GPU budget: ₹0 |
| Loss | 1.77 → **0.0072** in 160 steps (converged; a 250x drop) |

**This checkpoint:** step-164 draft of the first run. The official full 1-epoch run + router-accuracy eval is on the way and will replace this.

## Usage

The full agent runtime (skills, guardrails, execution loop, self-healing) lives in the open-source one-file repo — download `AURA_onefile.py` and run:

```bash
python AURA_onefile.py agent --model sarvamai/sarvam-2b --adapter ./aura_lora
```

To load the adapter yourself:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-2b", device_map="auto")
model = PeftModel.from_pretrained(base, "YOUR_USERNAME/aura-v1-lora")
tok = AutoTokenizer.from_pretrained("YOUR_USERNAME/aura-v1-lora")
```

> Note: the adapter expects the AURA system prompt (see `build_system_prompt()` in the one-file repo) — plain chat without it will still work but the skill-routing is trained for the agent format.

## Honest limitations

- 2B parameter model — simple tasks (files, launch, math, search) route reliably; complex multi-step chains are still learning
- English-only tokenizer inherited from the base model's vocab — Hindi is handled through transliteration-friendly Hinglish
- Not a hard security boundary — the sandbox stops accidents, not determined attackers

## Credits

Trained by **Jhalak Jhajhria** (13, India) — built in public on free compute.

Base model: [Sarvam-2B](https://huggingface.co/sarvamai/sarvam-2b) by Sarvam AI (Apache 2.0) · Trained with PEFT + Transformers · Compute: Kaggle T4

---
*AURA — flow. resonance. connection.* 👻
