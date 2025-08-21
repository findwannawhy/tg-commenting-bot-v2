import re
from typing import List, Pattern, Optional

_AD_REGEX: Optional[Pattern[str]] = None


def _compile_regex(phrases: List[str]) -> Pattern[str]:
	parts: List[str] = []
	for raw in phrases:
		p = (raw or "").strip()
		if not p:
			continue
		# escape regex meta and allow any whitespace in place of regular spaces inside the phrase
		escaped = re.escape(p)
		escaped = escaped.replace("\\ ", r"\\s+")
		parts.append(escaped)
	pattern = r"(" + "|".join(parts) + r")" if parts else r"$a^"  # never matches if empty
	return re.compile(pattern, re.I)


def set_ad_keywords(phrases: List[str]) -> None:
	global _AD_REGEX
	# If no phrases provided by user → disable heuristic (fall back to AI only)
	_AD_REGEX = _compile_regex(phrases) if phrases else None


def fast_heuristic_is_ad(text: str) -> bool:
	if not text:
		return False
	global _AD_REGEX
	# No user phrases configured → do not block by heuristic
	if _AD_REGEX is None:
		return False
	return bool(_AD_REGEX.search(text))
