# Junction 2026: all challenges (official, from the Junction Platform)

Captured 2026-11-13 ~19:45 from https://hackjunction.app/participate/45/challenges after the 19:30 unlock. There are 19 challenges.
Each section is a condensed version that keeps every concrete requirement, resource, criterion, prize and link. The platform has the full wording.

**How to enter (platform):** the project form's **Challenge Picker takes ONE challenge**. There's also a separate multi-select for "Tech Challenge Participation" (the stackable side prizes, listed at the bottom). Per #event-announcements: **submissions open Sat 14.11 12:00, deadline Sun 15.11 10:00.**

<!-- SUMMARY-TABLE -->

---

## 1. Decode the Noise (Hostinger)
**Task:** build an AI tool that detects propaganda, bots and bad actors online, and **deploy it live on Hostinger** with a public URL by the end of the hackathon. Detection logic and how you handle false positives count for more than a polished UI.
**Directions (pick or combine):**
- **Spot the swarm:** coordinated inauthentic behaviour (bot networks, astroturfing, brigading) from timing, account similarity, text reuse and amplification.
- **Name the technique:** classify manipulation techniques (loaded language, fear appeals, whataboutism, fabricated authority) and explain why in plain language.
- **Clean the traffic:** tell malicious bots, scrapers and credential stuffing apart from real visitors in web-server logs, with an actionable view for site owners.
- **Trust the source:** score provenance (domain age, cloned content, AI-generated text, link networks).
- They want transparent, measurable results: show precision and false-positive rate.

**Resources (free for 1 month):**
- Hostinger VPS KVM 2 per team: 2 vCPU, 8 GB RAM, 100 GB NVMe, root, Docker, free domain, one-click Ollama/n8n/Node/Python templates.
- **Hostinger AI Router credits**: one key for 150+ models (Anthropic, OpenAI, DeepSeek, Mistral, xAI…), compatible with the OpenAI and Anthropic SDKs.
- AI Builder plan.
- Hostinger Connector/API/CLI/MCP (works with Claude Code, Cursor): docs.hostinger.com/hostinger-connector, developers.hostinger.com, github.com/hostinger/api-mcp-server.

**Mentors:** Minijus Savickas (VPS), Emilis Strimaitis (AI agents) at the Hostinger booth. Talk Sat 13:00: "Vibe Coding Solved Building. Now Comes the Hard Part".
**Judging:** detection quality (accuracy, false-positive handling, explainability) **40%** · technical execution & live deployment on Hostinger 25% · originality & real-world impact 20% · demo clarity 15%.
**Prizes:** €2,000 / €1,000 / €500. All three also get 1 year of Hostinger hosting + domain, swag, and a social-media spotlight.

## 2. Earned Trust in AI Code (Tieto)
**Task:** design an **AI code-review concept** for agent-written code where a named human stays accountable. Define the concept, principles, a reference implementation and technology considerations (frontend, backend, security). Question whether line-by-line review is still right. It's a reference implementation, **not a product**.

**Each team must:**
- Define the recommended "PR review factory".
- State principles balancing risk, hallucination, transparency, confidence and quality.
- Illustrate it on **one sample agent-written change with its requirement ID**.
- Show what the AI found, **what it refused to decide and why**, and how the evidence lets a named human accept the change without a full read.
- Name the failure modes, and keep a decision log.
- Use synthetic or cleared data only.

**Deliverables:**
- A concept card: recommendation, principles, trade-offs, limits.
- A live walkthrough of where the trust comes from.
- A lessons log.
- A demo video or presentation.

Out of scope: a finished product, model training, vendor selection.

**Insight:**
- "AI reviews. Humans stay accountable." The AI review is advice with evidence, never the approval itself.
- Review depth follows risk.
- Trust is earned per area and withdrawn after escapes.
- The owner signs off on what was checked, what wasn't and why, and the residual risk they accept.

**Possible directions:**
- A generalist reviewer covering frontend, backend, security and performance.
- A classifier that separates code needing human review from code that meets the standard.

**Judges will ask:** What would make you distrust your own output? Who owns the decision? How does it scale?
**Resources:** Tieto's code standards (security, quality), a requirements set, sample agent-written changes with requirement IDs, and model access via partner credits on the platform.
**Judging:** Clarity 30% · Practicality 25% · Measurement 25% (reverts and incidents don't rise as review time falls, findings accepted, time to merge, share routed to humans) · Evidence 20%.
**Prizes:** €2,000 / €1,000 / €500.

## 3. Spread (Bahn Express)
**Task:** run a simulated **car-transport company for one week in Finland**.
- Dealers ask you and a competitor for a price. Quote too high and you lose the car.
- If you win, you post the drive as a gig in the drivers' app and set the fee. Pay too little and no driver takes it in time.
- Score = **gross margin** (dealer price minus driver pay). **At least 90% of cars must arrive by their deadline.**
- All teams play the same weeks on one **leaderboard**.

**Start:** create your team at **spread.bahnexpress.com/join** with event code **`spread-kmu9z6`**. The site has the handbook, rules, API reference, TypeScript and Python kits, and the leaderboard. Give your agent **spread.bahnexpress.com/llms.txt**.

**How you win:**
- Ranking rounds started Fri 19:30, on the same secret weeks for every team. Bahn's own system plays too; beat it.
- **Final Sun 08:00:** each team plays its own copy of one secret week (runs automatically, about 34 min). **Registration closes 07:30.**
- The top 10 by margin present to the jury (5–10 slides or a short video). The jury judges elegance, clarity, creativity and robustness; margin no longer counts after that.

Hints and questions at the Bahn booth.
**Prizes:** €2,000 / €1,000 / €500.

## 4. Break the Agent (AfterQuery)
**Task:** build a **reinforcement-learning environment** that today's best agents struggle in.
- It needs a clear goal, actions and a checkable reward.
- Run a frontier agent, and compare it with a **stronger reference run you can reproduce**.
- Show where and why the agent fails. Failures must come from real decisions, not withheld info.
- Make the reward hold up against **reward hacking**.

Optional stretch: train a smaller model on experience from your environment and show it improves (this is also the tiebreaker).
**Starting points:** multi-step planning, exploring an unfamiliar system, tool use, limited resources, adapting to new info. A small environment can be enough.
**Judging (points):**
- **The gap 30:** the frontier agent consistently scores below your reproducible reference run.
- **Meaningful difficulty 30:** you can explain the agent's mistakes, and it had the info and tools it needed.
- **Reward integrity 25:** doing nothing or stopping early earns ~0, and exploits don't beat real play.
- **Craft 15:** judges can run the evaluation from the README with one command.

Judging runs as a 10-min demo. Judges then rerun your reference run, 3 agent runs, a do-nothing baseline and a stop-early baseline, and spend 15 min trying to exploit the reward.
**Resources:**
- The **Harbor + OpenCode** agent harness, which connects to your environment over MCP.
- A scoring skeleton, **frontier-model API credits per team**, and a starter kit.
- Mentors Daksh Raghuwanshi and Ved Vedere (on the grand jury too).

**Prizes:** €3,000 + a **work trial at AfterQuery** (skip to the final interview stage) + merch / €1,500 + merch / €500.

## 5. Trusted Trade Agents (Valio Aimo)
**Task:** prototype the **interface and operating model** that lets an external **purchasing agent** order from Valio Aimo, a foodservice wholesaler serving 20k+ kitchens with 25k products from 700 suppliers. You build against **mocks** of today's webshop and backend, not the future marketplace. Use open, verifiable standards, not a single platform's black box.
**Core (must enable):**
1. Product, price and availability lookup in structured form.
2. Requesting price and availability, then **placing an order**, at fixed prices (no negotiation).
3. **Permission and human-approval management:** what the agent decides alone and what needs human sign-off.

**Optional:**
- Delivery time and sustainability comparison (certificates and labels; carbon ranges are a stretch).
- Substitution and exception handling. Last year's "Zero-Fail Logistics" covered this, so it isn't scored as core.
- Traceability and explainability.

**Example request:** "I need 1,000 litres of lactose-free milk by Monday…" A hotel chain's agent asks; you return structured products, prices and availability; the agent orders within its authority or routes the order to a human.
**Insight areas:** a trust protocol (minimum verification to trust stock, price and origin claims), a guardrails model (spend thresholds, substitutions outside preferences), response logic (partial availability, good-better-best substitution), and an audit trail.
**Scenario ideas:**
- Need → recipe → order.
- A last-minute event.
- A supplier stock-out with a recipe-compatible substitute.
- A proactive selling agent.
- Allergen-safe substitution (schools, hospitals).
- Detecting drift in standing orders.

**Resources:**
- A synthetic catalog, simulated suppliers and **mock APIs**: prices, availability behaviour, lead times, certificates.
- Mentoring.
- Preferred tech: agentic/tool-use frameworks, open interop standards, recommendation logic, optional verifiable credentials. x402/Nevermined payment standards are relevant but not required.

**Access (posted on the platform for all participants):**
- Portal: **junction26.valioaimo.fi**, password on the platform (datasets, API docs, mock-backend instructions).
- Mock-API key: `junct_tDnQn4VI_nfhELGCOEkY4vbuxm324MmYmILuokYG03Y`.
- Webshop test accounts will be shared later.

**Mentors:** Hanna Forsström, Pirjo Kaartinen (marketplace), Minni Ruhtinas (online sales), Tommi Nurmi-Aro, Ville Takanen, Riku Tapper (IT/data governance).
**Judging:** **working end-to-end demo 30%** (request → lookup → order/approval) · solving the user need 20% · autonomy & controls 20% · robustness to exceptions 15% · openness & credible path to production 15%.
**Prizes:** **€4,000** / €2,000 / €1,000.

## 6. The Internet Butler / Your Personal Assistant (Exa)
**Task:** a **personal agent that takes tedious tasks off people's hands**. It finds what matters across the web with Exa, works out what needs to happen, and gets the task as close to done as possible, leaving final approval to the user. **Demo it on a real case.**
**Examples:** checking whether a rent increase is legal, which benefits a family qualifies for, planning special occasions, errands, business info, travel disruption.
**Resources:**
- **$50 Exa credits per participant, code `EXA50JUNCTION`** (redeem at dashboard.exa.ai/billing).
- APIs: Search (fast, balanced and deep modes), Contents (full text, highlights, summaries, subpage crawling), filters (date, domain, category), structured JSON output, past versions of pages.
- Websets, Monitors, a hosted MCP server, Python and JS SDKs, and examples at demos.exa.ai.
- Exa engineers on site, plus the Discord channel.

**Judging (20% each):** use of Exa for info beyond training data · usefulness to a specific group of real people · clarity in showing sources · originality · the demo working on **live queries, including the judges' own**.
**Prizes:**
1. $2,500 + $5,000 Exa credits + hoodies + an Exa TRMNL dashboard.
2. $1,500 + $3,000 credits + hoodies + a macropad.
3. $500 + $1k credits + hoodies.

## 7. The Green European Renaissance (Arduino) (Hardware track)
**Task:** use **Edge AI on the Arduino UNO Q** to build physical AI for local problems that make Europe greener and more human. Examples: local flora and fauna, saving energy, independence for elderly people, industrial impact, pollution, livable cities.
**Submission:** besides the Junction platform, you **must publish on Arduino Project Hub** to be eligible. Go to projecthub.arduino.cc → + NEW PROJECT → tag **"Junction Finland 2026"**.
**Resources:** an Arduino UNO Q kit with Modulino sensors from the Arduino Hardware Lab on the 5th floor. Install Arduino App Lab (arduino.cc/en/software/#app-lab-section).
**Judging:** technical implementation & edge AI 30% · environmental & social impact 30% · creativity 20% · working physical prototype & demo 20%.
**Prizes (per team member):** 1st a Ventuno Q unit · 2nd an UNO Q 4G + Plug and Make Kit · 3rd a Plug and Make Kit.

## 8. AI Glasses for Everyone, Everyday (Meta)
**Task:** build the best **everyday use case for Meta AI glasses**, showing how a **first-person field of view** enables new or better interactions. It has to deliver value beyond mirroring a phone screen and be **inclusive for people with all abilities**.
- Articulate the problem and audience clearly. Deliver a **functional prototype + deck**.
- **Display and audio-first experiences count equally**, as do **web apps and Device Access Toolkit (DAT) mobile integrations**.
- The prototype should handle **permissions, errors, performance trade-offs, connectivity and device-state changes**.
- Include appropriate consent and safe data practices in the demo. Show a plausible path to adoption or a pilot.

**Bonus:** the best teams may be invited to Meta's **AI forum on Wed 18 Nov 2026** (future of wearable AI).
**Resources:**
- **10× Meta Ray-Ban Display + 10× Ray-Ban Meta Gen 2**, shared, borrowed through the hardware desk. Display loans are limited to 120 min and Gen 2 to 240 min.
- Charging and hygiene kits.
- Developer support at the Meta booth. Challenge contact: **Amanda Le**.
- Mentors: Ming Lin, William La, Amy Lee, Odil Nikadambaev (web apps), Dennis Punjabi (DAT iOS), Amanda Le (DAT Android).

**Judging (25% each):** creativity & use case · technical execution · real-world value & feasibility · user experience (input, accessibility, trust).
**Prize:** each member of the winning team gets **Ray-Ban Meta glasses**.

## 9. Stop Simulating, Start Moving: See It. Detect It. Act on It. (Ignite Next x Intel)
**Task:** turn a real robot arm into an **autonomous inspection cell** in one weekend:
1. **Teleoperate** the arm to record demonstrations.
2. **Fine-tune a vision-language-action (VLA) policy** in **Physical AI Studio** on cloud GPUs.
3. In parallel, train an **Anomalib** model that tells good Duplo bricks from defective ones (you mark the defects yourself).
4. Combine them: the arm picks each brick, shows it to the inspection camera and drops it in the "good" or "defective" bowl, **fully autonomously**. Models are optimized with **OpenVINO** and run on the laptop's **CPU, GPU and NPU**.

Stretch: faster cycles, messier piles, new defect types.
**Insight:** they want to know how few demos are enough, robustness to bricks in new positions, and full-loop speed on CPU/GPU/NPU. Ideas they'd like: use the anomaly score to trigger a re-grasp, share cameras between policy and inspection, quantize with OpenVINO.
**Station per team (25 total, one per team):**
- An Intel Core Ultra Series 3 (Panther Lake) laptop with 64 GB RAM.
- An **SO-101 follower arm** with a wrist camera, plus a **leader arm** for teleoperation.
- **2 external cameras** (top view and inspection; at least one is a RealSense D415 depth camera).
- Duplo bricks, bowls, markers.
- **Cloud GPU instances** for policy training.
- Physical AI Studio, Anomalib, OpenVINO + OpenVINO Physical AI, and Intel's Robotics AI Suite.

**Support:** a hands-on session **Sat 12:00**. 3–5 Intel robotics and edge-AI engineers are on site every day.
**Contact:** Marvin Müller (Ignite Next) (email on the platform). On Discord: Rajesh | intel ("come to the 5th floor challenge area").
**Challenge Terms** (you agree to them by participating): drive.google.com/file/d/1nKs-e-4FNk-AoMq1fCeRKM_6mpIkArRt/view
**Judging:**
- Task success 30% (accuracy, cycle time, robustness to shifted or cluttered bricks).
- **Intel tech integration 30%** (OpenVINO Physical AI + Robotics AI Suite, offloading to NPU and iGPU).
- Dataset & pipeline quality 20% (clean teleop data, fine-tuning in Physical AI Studio).
- System efficiency 20% (latency, thermal headroom, power on Core Ultra).

**Prizes:** **€5,000** / €3,000 / €2,000.

## 10. The Robot Teleoperation Challenge (Rokoko)
**Task:** use **Rokoko motion-capture suits and gloves** to control robotic hands, arms or humanoids, physically or in simulation. Build something useful, unexpected or unbelievable.
**Interests:**
- Precise manipulation (fragile objects, tools, liquids).
- Collecting teleop demonstrations to train autonomous policies.
- Creative or entertainment uses.

**Resources:**
- Smartgloves II, Smartsuit Pro II, Coil Pro, and the Device SDK.
- Access to the **Unitree G1** humanoid with dexterous hands, coordinated with Aalto's challenge. Possibly a Booster T1.
- An Inspire robotic-hand integration from Aalto Robotics Club.

**Access:**
1. Create a Rokoko ID.
2. Send your email and ID to Rokoko through their form (rokoko.com/mocap/junction-hackathon).
3. Get accepted.
4. Activate the SDK license.

**Mentors:** Victoria Stein (Rokoko) and Aalto Robotics Club.
**Judging:** technical execution 35% · originality 25% · usefulness & impact 25% · demonstration of human-controlled robotics 15%.
**Prizes:** Smartgloves II + SDK license + blog feature (~€4,990) / Smartsuit Pro II (€2,295) / Smartgloves I + merch (€1,095).

## 11. Aaltonomous Humanoid (Aalto University)
**Task:** make Aalto's **Unitree G1 "Väiski"** **sort objects into boxes autonomously** within 2 minutes.
- **Setup:** the robot stands behind a counter with its legs and waist stiff, using two 7-DoF arms and Dex3 three-finger hands. 3–5 printed objects (cubes, cylinders, prisms, ~4.5 cm, red/yellow/blue) lie in random poses. There are 3 identical orange boxes, left/centre/right, 40 cm away.
- **Fixed sorting:** red → left, yellow → centre, blue → right.
- **Sort by instruction:** natural-language instructions with unseen phrasings.
- **Stretch:** put each object through the matching hole in a lid.
- **Interface:** the policy gets the head RGB-D camera, both wrist cameras, joint states and the instruction. It outputs joint targets at **30 Hz**.

**How it works:**
- Develop in a ready-made **Isaac Sim** scene on Verda GPUs. Use an RTX-class GPU with at least 24 GB (it won't render on A100/H100); use on-demand instances and a persistent volume.
- Real-robot datasets will be published on Hugging Face during the event (links on Discord).
- **Submit a Docker image** policy server. Finalists are picked in simulation on an unseen scene, the **top ~5 run on the real G1 on Sunday**, and the best real-robot score wins.

**Safety:** the robot only runs code that passes simulation (no crashes, within limits, ≥27 commands/s).
**Starter repo:** **github.com/zupermiez/aaltonomous-humanoid** (aaltonomous/cloud/QUICKSTART.md, docs/participants/). Simulator commands: `aalto sim`, `aalto check`, `aalto baseline`, `aalto eval --policy ws://localhost:8000`.
**Mentors:** Erkka Lehtonen, Viktor Solianoi, and ~30 Aalto Robotics Club volunteers.
**Judging:** **real-world result 40%** · generality 25% · originality 20% · presentation 15%.
**Prizes:** €2,000 + a visit to the Aalto robotics lab. The top ~5 run live on the real robot.

## 12. Make Unicorns Real (Cala)
**Task:** build a product on **Cala's structured, sourced public data** (companies, people, funding rounds, investors, and the connections between them) that solves **one real need for founders and/or investors**. Focus beats breadth. Aim for something usable at **Slush**, which starts right after.
**Ideas:**
- Finding the investors who actually lead rounds like yours and are still active.
- Reading a market.
- Spotting young companies early.
- Quick sourced background checks.
- Finding first customers among companies that just raised.
- Market sizing, mapping acquirers.

**Resources:**
- API + MCP, docs.cala.ai.
- Credits: console.cala.ai → Billing → Extra credits → Add promo code **`MAKEUNICORNSREAL`** (no card needed).
- Workshop **Sat 11:00**.
- Mentors: Issey Masuda (CTO) and Cala engineers. Contact: Elies Delgado (email on the platform).

**Judging (25% each):** real need · execution (works end to end) · originality · use of Cala as the core.
**Prizes:** each winner becomes a "Cala Comet" (~€5k per team of 4: one hackathon per month for 3 months, travel and accommodation paid) + €1,000 Cala credits + merch. Runner-up gets €500 credits + merch.

