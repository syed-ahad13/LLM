"""The adapters crawl.py and report.py can use, by site id."""

from collect.scrape.sites.base import SiteAdapter
from collect.scrape.sites.humsub import HumSub
from collect.scrape.sites.voa_urdu import VoaUrdu

ADAPTERS: dict[str, type[SiteAdapter]] = {a.site_id: a for a in (VoaUrdu, HumSub)}
