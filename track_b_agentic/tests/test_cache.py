import time

from app.cache import TTLCache


def test_hit_and_miss_are_counted():
    cache = TTLCache(max_entries=4, ttl_seconds=60)
    assert cache.get("absent") is None
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert cache.stats() == {"size": 1, "hits": 1, "misses": 1}


def test_entries_expire():
    cache = TTLCache(max_entries=4, ttl_seconds=0.05)
    cache.set("k", "v")
    time.sleep(0.08)
    assert cache.get("k") is None
    assert cache.stats()["size"] == 0


def test_cache_is_bounded_and_evicts_least_recently_used():
    cache = TTLCache(max_entries=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")           # "a" becomes most recently used
    cache.set("c", 3)        # evicts "b"
    assert cache.get("b") is None
    assert cache.get("a") == 1 and cache.get("c") == 3
    assert cache.stats()["size"] == 2
