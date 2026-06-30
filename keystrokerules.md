assume that for each group, it is intended for the associated region, then its subsequent listings to be the appropriate keystroke, then the requirement based on state transition. from recast states, assume that if no state change occurs within 3.1 seconds afterwards of first recast detection, no keystroke was inputted. if any state turns into "disabled" or "missing", do not input any keystroke

"Q":
  - "Q1": "ready" -> any state
  - "Q2": "recast" -> anystate besides "disabled" or "missing"

"W":
  - "W": "ready" -> "recast"

"E":
  - "E": "ready" -> anystate besides "disabled" or "missing"

"R":
  - "R": "ready" -> "cooldown"

"D":
  - "cleanse": "ready" -> "cooldown"

"F":
  - "flash": "ready" -> "cooldown"

"prowler":
  - "prowler": "ready" -> "cooldown"

"rocketbelt":
  - "rocketbelt": "ready" -> "cooldown"

"ward":
  - "ward": "ready" -> "cooldown"
