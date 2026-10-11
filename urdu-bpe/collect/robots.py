"""robots.txt parsing and matching (RFC 9309), for the downloader and the scraper.

The stdlib urllib.robotparser does plain prefix matching, so it ignores the `*`
and `$` wildcards that sites we use rely on (LINDAT, Mendeley, GitHub). Rules:

- Group selection: groups whose user-agent equals our product token
  (case-insensitive) apply; otherwise the `*` groups. Matching groups merge.
- Matching: `*` matches any run of characters and a trailing `$` anchors the
  end. The longest matching rule wins; Allow wins a tie. No match means allowed.
- /robots.txt itself is always allowed.
- Crawl-delay (non-standard) is parsed so the client can honour it.
"""

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Rule:
    allow: bool
    pattern: str
    regex: re.Pattern[str]

    @classmethod
    def compile(cls, allow: bool, pattern: str) -> "Rule":
        anchored = pattern.endswith("$")
        body = pattern[:-1] if anchored else pattern
        regex = ".*".join(re.escape(part) for part in body.split("*")) + ("$" if anchored else "")
        return cls(allow, pattern, re.compile(regex))


@dataclass
class Group:
    agents: list[str] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    crawl_delay: float | None = None


class RobotsTxt:
    def __init__(self, text: str = "", *, disallow_all: bool = False) -> None:
        self.disallow_all = disallow_all
        self.groups = _parse(text)

    @classmethod
    def unreachable(cls) -> "RobotsTxt":
        """RFC 9309 2.3.1.4: if robots.txt cannot be fetched (5xx, network), assume complete disallow."""
        return cls(disallow_all=True)

    def _groups_for(self, agent: str) -> list[Group]:
        token = agent.lower()
        named = [g for g in self.groups if any(a.lower() == token for a in g.agents)]
        return named or [g for g in self.groups if "*" in g.agents]

    def allowed(self, url: str, agent: str) -> bool:
        parts = urlsplit(url)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        if target == "/robots.txt":
            return True
        if self.disallow_all:
            return False
        best: Rule | None = None
        for group in self._groups_for(agent):
            for rule in group.rules:
                if rule.regex.match(target) and (
                    best is None
                    or len(rule.pattern) > len(best.pattern)
                    or (len(rule.pattern) == len(best.pattern) and rule.allow)
                ):
                    best = rule
        return best is None or best.allow

    def crawl_delay(self, agent: str) -> float | None:
        delays = [g.crawl_delay for g in self._groups_for(agent) if g.crawl_delay is not None]
        return max(delays) if delays else None


def _parse(text: str) -> list[Group]:
    groups: list[Group] = []
    current: Group | None = None
    in_agent_lines = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        key, sep, value = line.partition(":")
        if not sep:
            continue  # blank, comment-only or malformed (e.g. "Allow /api/docs")
        key, value = key.strip().lower(), value.strip()
        if key == "user-agent":
            if current is None or not in_agent_lines:
                current = Group()
                groups.append(current)
            current.agents.append(value)
            in_agent_lines = True
            continue
        in_agent_lines = False
        if current is None:
            continue  # rules before any user-agent line apply to no one
        if key in ("allow", "disallow") and value:
            current.rules.append(Rule.compile(key == "allow", value))
        elif key == "crawl-delay":
            try:
                current.crawl_delay = float(value)
            except ValueError:
                pass
    return groups
