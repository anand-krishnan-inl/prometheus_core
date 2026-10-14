---
title: Day 2 Kickoff
footer: Day 2 · Kickoff
accent: team
index:
  time: Day 2
  heading: Day 2 kickoff
  color: var(--crimson)
  blurb: 'A short refresher before building: the two goals, the Hollowmere almanac, the steps from template to a working agent, teamwork in git, what to do if you finish early, breakout rooms.'
---

# Day 2 kickoff {.title kicker="Agent developer hackathon · Day 2" pills="All teams | 10 min | then breakout rooms"}
Your platform is up and running. Today every team turns it into a Table & Plot agent: a quick refresher, then off to the breakout rooms.

---

## One agent, two skills {eyebrow="The goal"}

:::: grid 2 gap=30px
::: card acc tag="Skill 1"
### Extract a table

{.body}
From an uploaded markdown document, find the table the user describes **by its caption**, parse it, clean the numbers, and publish it as a **table artifact**.
:::

::: card acc tag="Skill 2"
### Plot it

{.body}
From a table artifact **or** an uploaded CSV, draw the plot the user asked for as a **PNG artifact**. No columns given? **Ask**, offering the real column names.
:::
::::

::: callout
The orchestrator chains them: skill 1 runs, returns the table's id and column names, and the orchestrator passes them to skill 2. You design the skills; the orchestrator decides when to call each.
:::

---

## Your practice file: the Hollowmere almanac {eyebrow="What you get"}

:::: cols 1.3fr 1fr gap=26px
{.tiny}
`hollowmere_almanac.md` (excerpt, as rendered)

{.small}
**Table 2. Monthly rainfall and wild mushroom harvest in the Whispering Wood for the year 2023, with the average market price fetched at the Mossbury Saturday market.**

{.sm}
| Month | Rainfall (mm) | Chanterelle yield (kg) | Porcini yield (kg) | Average price (copper per kg) |
|---|---|---|---|---|
| January | 89.4 | 9.7 | 7.5 | 34.6 |
| February | 96.0 | 6.8 | 6.7 | 35.2 |
| … | … | … | … | … |
| December | 54.9 | 7.2 | 3.7 | 38.3 |

|||

{.sm}
- Three captioned tables: the gnome **census**, the mushroom **harvest**, the **dragon** register
- Bold caption directly **above** each table, prose in between
- Units in the headers, text columns mixed with numbers

#### Try prompts like {.acc}

::: callout good
*"Extract the table with the mushroom harvest and plot the porcini yield versus month."*
:::

{.small}
Write your own test files too, with other captions, units, and symbols. An agent that only knows this file isn't much fun.
::::

---

## From template to a working agent {eyebrow="The steps"}

:::: grid 2 gap=30px
::: steps
#### Scaffold the team agent

{.small}
`/create-agent` in Claude Code, or `setup_agent.py` from the agent template. You pick the agent's name and the repo's name.

#### Two stub skills, routed

{.small}
Add both skills to `agent_card.json` and a dispatch on `params["skill_id"]`. Upload the almanac and check that the orchestrator calls both, in order.

#### Walking skeleton

{.small}
Extract emits a real table artifact; plot emits a PNG via DeepLynx. Crude is fine: get it working end to end first.

#### Make it robust

{.small}
Caption and column matching (an LLM helps), asking for columns, the CSV path, your own test files, tests that pass.
:::

|||

::: card acc
### Keep these open

{.sm}
- Deck `06`: the challenge, card JSON, implementation hints (appendix, key `A`)
- `cheatsheet.html`: commands and helpers
- `/how`, `/diagnose`, `/create-agent` in Claude Code
- Mentors in the room
:::

::: callout warn
Get the practice prompt working end to end in the **first two hours**, then make it robust.
:::
::::

---

## What if we finish early? {eyebrow="Be creative"}
> Once extract and plot work, the floor is yours. Surprise us.

:::: grid 3 gap=22px
::: card acc tag="Grow"
### Expand your skills

{.sm}
- Customize plot colors, styles, log scales
- Plot several series, or several graphs in one figure
- Add skills: list the tables, describe a table, convert units
:::

::: card acc tag="Build"
### Add another agent

{.sm}
- Dealer's choice: anything that fits your team's interests
- A second agent the orchestrator can chain with your first
- Same recipe: template, card, tests
:::

::: card acc tag="Fix"
### Improve the platform

{.sm}
- Hit a platform issue? **Write it down**: what you did, what you saw
- Tell a mentor and add it to your README
- Brave? Take a stab at a fix in a PR
:::
::::

::: callout
Whatever you build, do it in small PRs. Your extras are part of the story you tell at the end of the day.
:::

---

## Teamwork in git {eyebrow="Today's other goal"}
> The agent is half the fun. The other half is practicing the way we work: repos, forks, and reviewed pull requests.

::: flow
- [violet] **Owner** | ~gh repo create
- -> fork {w=70}
- [sky flex=1.1] **Teammates** | ~gh repo fork --clone
- -> branch {w=80}
- [orange] **Feature branch** | small commits
- -> PR {w=60}
- [green flex=1.1] **Review & merge** | a teammate approves
:::

:::: grid 3 gap=22px
::: card acc
#### Make the repo

{.small}
One teammate scaffolds the agent, then `git init` and `gh repo create <repo> --source=. --push`. Pick any name you like.
:::

::: card acc
#### Fork and branch

{.small}
Everyone else runs `gh repo fork <owner>/<repo> --clone`, works on a `feature/…` branch, and runs `git pull --rebase upstream main` often.
:::

::: card acc
#### Review each other

{.small}
Open small PRs to the owner's repo. The owner reviews teammates' PRs; a teammate reviews the owner's. Comment, ask, approve.
:::
::::

::: callout good
At the end of the day, each team **shows its repo**: the agent, the PRs, and the reviews. No scores, just show and tell.
:::

---

# Breakout rooms {.section num="→"}
We'll now assign the eight teams to their breakout rooms. Happy hacking!
