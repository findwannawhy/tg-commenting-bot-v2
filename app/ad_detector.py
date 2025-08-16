import re


AD_LINKS = re.compile(r"https?://|t\.me/|@\w+", re.I)
AD_PROMO = re.compile(
    r"промокод|скидк|распродаж|подписывайтесь|подпишись|подписка|"
    r"реклама|#ad|#реклама|"
    r"рекомендую|в\s*закрепе|ссылка\s+в|по\s+ссылке|"
    r"переходи(?:те)?",
    re.I,
)
AD_CONTACTS = re.compile(r"\+?\d[\d\-\s]{7,}", re.I)
AD_COMMERCE = re.compile(r"купить|продаж|заказ|доставка|цена|руб\.|₽|\$", re.I)


def fast_heuristic_is_ad(text: str) -> bool:
	if not text:
		return False
	if AD_LINKS.search(text):
		return True
	if AD_PROMO.search(text):
		return True
	if AD_CONTACTS.search(text):
		return True
	if AD_COMMERCE.search(text):
		return True
	return False
