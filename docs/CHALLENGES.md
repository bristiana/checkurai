# Junction 2026: side challenges

Our main challenge is Intel Physical AI (see [PLAN.md](PLAN.md)). Every other team member can submit to one more challenge, which gives us more chances to win. This file lists every challenge on hackjunction.app (event 45). For each one it notes how well it fits us, how much competition to expect and whether it can be built in the time left.

**Time left:** the final submission is due **Sun 10:00** (Helsinki). We aim to submit by 09:30. From Saturday evening, that leaves about 12 hours.

**Rules for picking:**
- Software only. There is no time left to get hardware or access to hardware.
- Pick niche challenges over crowded ones. A solid project in a field of 5 teams has better odds than a great one in a field of 60.
- Read the judging criteria, not the title. Some challenges score the concept more than the code.
- Each side project gets **its own repo**. This repo turns public for the Intel submission.

Legend:
- **Fit**: how well the challenge suits our skills (CV, detection, Python, agents).
- **Crowd**: how many teams we expect to compete.
- **12 h?**: whether a winning-quality project can be built in about 12 hours.

## Shortlist

| # | Challenge | Partner | Prize (1st / 2nd / 3rd) | Why it's on the list |
|---|---|---|---|---|
| 1 | **Earned Trust in AI Code** | Tieto | €2000 / €1000 / €500 | The concept counts more than the code ("more weight for the concept"). The sample repo and branches are provided, so there is no data work. We have a real story to tell: our Intel code is written by AI agents and approved by humans. |
| 2 | **Trusted Trade Agents** | Valio Aimo | €4000 / €2000 / €1000 | The largest software-only prize. The core is clear: product lookup, ordering, and permissions with human approval. It can be built as an MCP server, a mock catalog and an approval UI. It will probably be crowded. |
| 3 | **Make Blueprints Talk** | Stoa | €1000 / €750 / €250 + work trial | It suits our CV background, and its track has only this one challenge, so the crowd should be small. It is technically hard to do well in 12 hours. |

### 1. Tieto: how to make it concrete
- **Change to review:** use `ju26-issue-2283` or `ju26-issue-2234` (4 files each) from https://github.com/te-industry-tech-office/junction26-messagepack-csharp. Fork it first.
- **Output:** a "trust dossier" posted on the PR. It covers:
  - what changed and why, linked to the requirement ID;
  - the blast radius;
  - a confidence score for each claim, with evidence (tests run, call sites checked);
  - an explicit list of what the AI **refused to decide**, with the reason;
  - a sign-off by a named human.
- **Deliverables:** a concept card (principles, trade-offs, limits), a live walkthrough, a lessons and decisions log, and a demo video.
- **Optional stretch:** run the same dossier on the 237-file branch (`ju26-issue-2297`). It shows where the concept breaks, and naming failure modes is part of the brief.

### 2. Valio Aimo: how to make it concrete
- An MCP server or REST API with these tools: `search_products`, `get_price_availability`, `place_order`.
- A policy file that sets what the agent can order alone (spending limit, approved categories, substitution rules). Anything outside it goes to a human approval queue.
- A trace log of every agent action and the reason for it.
- **Demo:** the brief's own example, "1,000 L lactose-free milk by Monday".

## All challenges

| Track | Challenge | Partner | Prize 1st | Fit | Crowd | 12 h? | Note |
|---|---|---|---|---|---|---|---|
| Agentic AI | Decode the Noise | Hostinger | €2000 | med | high | yes | Must be deployed live on Hostinger. Judged on honest detection and false-positive handling. |
| Agentic AI | **Earned Trust in AI Code** | Tieto | €2000 | high | med | **yes** | Shortlist #1. |
| Agentic AI | Make Unicorns Real | Cala | ~€5000 in trips | med | high | yes | Slush-themed, so it will be crowded. Cala's data must be core to the product. |
| Agentic AI | Spread | Bahn Express | €2000 | med | med | risky | A leaderboard game where other teams have tuned since Friday. Final registration closes Sun 07:30. |
| Agentic AI | Break the Agent | AfterQuery | €3000 + work trial | med | med | risky | Judges re-run everything and hunt for reward exploits for 15 min. It is hard to make that robust in 12 h. |
| Agentic AI | **Trusted Trade Agents** | Valio Aimo | €4000 | high | high | **yes** | Shortlist #2. |
| Agentic AI | Internet Butler | Exa | $2500 + credits | med | very high | yes | A very generic idea; expect many teams. |
| Hardware | Green European Renaissance | Arduino | Ventuno Q per member | low | med | no | Needs an Arduino UNO Q. |
| Hardware | AI Glasses | Meta | Ray-Ban Meta per member | low | med | no | Needs the glasses. |
| Hardware | Intel Physical AI | Ignite Next × Intel | €5000 | — | — | — | **Our main challenge.** |
| Hardware | Robot Teleoperation | Rokoko | Smartgloves (~€5k) | low | low | no | Needs the mocap suit. |
| Hardware | Aaltonomous Humanoid | Aalto | €2000 + lab visit | med | low | no | Isaac Sim plus a Docker policy. Too heavy for 12 h. |
| Healthcare | Equal Care Across Finland | AstraZeneca | €1000 | med | low | yes | Needs public Finnish cancer data and a "Goal 3" indicator. Every claim needs a source. |
| Healthcare | Beyond the Clinic | Aalto | €2000 + lab | low | low | yes | Home maintenance after TMS depression treatment. Mostly concept and app. |
| AI Infra | **Make Blueprints Talk** | Stoa | €1000 + work trial | high | low | hard | Shortlist #3. |
| Data | Web Is Your Real-Time DB | Context.dev | €2000 | med | high | yes | Open-ended, so crowded. |
| Data | Rethinking Recruitment | City of Espoo | €1500 | low | med | yes | The brief is vague, so judging will be subjective. |
| Data | Wikipedia Through Time | MariaDB | €1000 + certs | med | low | yes | The SVT (time-travel) track gets extra credit and few teams will try it. A good niche backup. |
| FinTech | Personal Economic Trainer | S-Pankki | €2000 Airbnb | med | high | yes | Uses a synthetic bank dataset. A mobile-first chat app over your own spending. |

## Who takes what

| Person | Challenge |
|---|---|
| Bogdan | Intel (lead) |
| Marius | Intel (laptop, code) |
| Cristiana | **Exa: The Internet Butler** |
| Quentin | _TBD_ |

Check first that you are not needed for the Intel teleop demos (PLAN.md §10, Sat 09:00–12:00) before you commit to a side challenge.
