# xenobot-organism-designer

AI platform for designing and simulating programmable living machines.

---

## What Are Xenobots?

In 2020, researchers at Tufts University and the University of Vermont
took stem cells from frog embryos, placed them in a petri dish, and
watched them reorganize into entirely new living structures — without
any genetic modification. These **xenobots** could move, push objects,
heal when damaged, and in 2021, reproduce by gathering loose cells and
compressing them into new organisms.

They are the first living machines designed by AI. Potential
applications include clearing arterial plaque, removing microplastics
from water, and targeted drug delivery. Building one today requires
supercomputing infrastructure and months of manual lab work.
This platform removes that barrier.

*Original research: Kriegman et al. 2020, 2021 — PNAS*

---

## What This Does

Input a task in natural language. Get back:

- An AI-optimized xenobot morphology (NEAT evolution → Bayesian 
  optimization → gradient ascent)
- A physics simulation matched to your fluid environment
- A wet lab assembly protocol
- A feedback loop that ingests your experimental results and 
  refines the design

---

## Quickstart
```bash
git clone https://github.com/yourname/xenobot-organism-designer
cd xenobot-organism-designer
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
uvicorn backend.api.main:app --reload --port 8000
# open frontend/index.html
```

Requires Python 3.10+ and an [Anthropic API key](https://console.anthropic.com).
