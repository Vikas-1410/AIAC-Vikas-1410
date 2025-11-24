"""LRU Cache for Product Pages with Metrics and Tests"""
from typing import Optional, Dict, Any
from dataclasses import dataclass
from time import time
import unittest

@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    cache_size: int = 0
    max_size: int = 0
    @property
    def hit_rate(self) -> float:
        return (self.hits / self.total_requests * 100) if self.total_requests else 0.0
    @property
    def miss_rate(self) -> float:
        return (self.misses / self.total_requests * 100) if self.total_requests else 0.0

class LRUNode:
    def __init__(self, key: str, value: Any):
        self.key, self.value = key, value
        self.prev = self.next = None
        self.timestamp = time()

class LRUCache:
    def __init__(self, capacity: int = 100):
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        self.capacity = capacity
        self.cache: Dict[str, LRUNode] = {}
        self.metrics = CacheMetrics(max_size=capacity)
        self.head = LRUNode("", None)
        self.tail = LRUNode("", None)
        self.head.next = self.tail
        self.tail.prev = self.head
    def _add_node(self, node: LRUNode):
        node.prev, node.next = self.head, self.head.next
        self.head.next.prev = node
        self.head.next = node
    def _remove_node(self, node: LRUNode):
        node.prev.next, node.next.prev = node.next, node.prev
    def _move_to_head(self, node: LRUNode):
        self._remove_node(node)
        self._add_node(node)
    def _pop_tail(self) -> Optional[LRUNode]:
        last = self.tail.prev
        if last == self.head:
            return None
        self._remove_node(last)
        return last
    def get(self, key: str) -> Optional[Any]:
        self.metrics.total_requests += 1
        node = self.cache.get(key)
        if node is None:
            self.metrics.misses += 1
            return None
        self._move_to_head(node)
        self.metrics.hits += 1
        return node.value
    def put(self, key: str, value: Any) -> bool:
        node = self.cache.get(key)
        if node is None:
            new_node = LRUNode(key, value)
            if len(self.cache) >= self.capacity:
                tail = self._pop_tail()
                if tail:
                    del self.cache[tail.key]
                    self.metrics.evictions += 1
            self.cache[key] = new_node
            self._add_node(new_node)
            self.metrics.cache_size = len(self.cache)
            return True
        else:
            node.value, node.timestamp = value, time()
            self._move_to_head(node)
            return False
    def delete(self, key: str) -> bool:
        node = self.cache.get(key)
        if node is None:
            return False
        self._remove_node(node)
        del self.cache[key]
        self.metrics.cache_size = len(self.cache)
        return True
    def clear(self):
        self.cache.clear()
        self.head.next, self.tail.prev = self.tail, self.head
        self.metrics.cache_size = 0
    def size(self) -> int:
        return len(self.cache)
    def get_metrics(self) -> CacheMetrics:
        self.metrics.cache_size = len(self.cache)
        return self.metrics

class ProductPageCache:
    def __init__(self, capacity: int = 100):
        self.cache = LRUCache(capacity)
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(product_id)
    def cache_product(self, product_id: str, product_data: Dict[str, Any]) -> bool:
        return self.cache.put(product_id, product_data)
    def get_stats(self) -> Dict[str, Any]:
        m = self.cache.get_metrics()
        return {'size': m.cache_size, 'capacity': m.max_size, 'hits': m.hits,
                'misses': m.misses, 'evictions': m.evictions,
                'total_requests': m.total_requests,
                'hit_rate': f"{m.hit_rate:.2f}%", 'miss_rate': f"{m.miss_rate:.2f}%"}

class TestLRUCache(unittest.TestCase):
    def setUp(self):
        self.cache = LRUCache(capacity=3)
    def test_basic_ops(self):
        self.cache.put("k1", "v1")
        self.cache.put("k2", "v2")
        self.assertEqual(self.cache.get("k1"), "v1")
        self.assertIsNone(self.cache.get("k3"))
    def test_eviction(self):
        for i in range(4):
            self.cache.put(f"k{i}", f"v{i}")
        self.assertIsNone(self.cache.get("k0"))
        self.assertEqual(self.cache.get("k3"), "v3")
    def test_lru_order(self):
        self.cache.put("k1", "v1")
        self.cache.put("k2", "v2")
        self.cache.put("k3", "v3")
        self.cache.get("k1")
        self.cache.put("k4", "v4")
        self.assertIsNone(self.cache.get("k2"))
        self.assertEqual(self.cache.get("k1"), "v1")
    def test_update_no_eviction(self):
        for i in range(3):
            self.cache.put(f"k{i}", f"v{i}")
        self.cache.put("k0", "updated")
        self.assertEqual(self.cache.size(), 3)
        self.assertEqual(self.cache.get("k0"), "updated")
    def test_delete_clear(self):
        self.cache.put("k1", "v1")
        self.cache.put("k2", "v2")
        self.assertTrue(self.cache.delete("k1"))
        self.cache.clear()
        self.assertEqual(self.cache.size(), 0)

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.cache = LRUCache(capacity=3)
    def test_hit_miss(self):
        self.cache.put("k1", "v1")
        self.cache.get("k1")
        self.cache.get("k2")
        m = self.cache.get_metrics()
        self.assertEqual(m.hits, 1)
        self.assertEqual(m.misses, 1)
        self.assertAlmostEqual(m.hit_rate, 50.0, places=1)
    def test_eviction_count(self):
        for i in range(4):
            self.cache.put(f"k{i}", f"v{i}")
        self.assertEqual(self.cache.get_metrics().evictions, 1)
        self.cache.put("k4", "v4")
        self.assertEqual(self.cache.get_metrics().evictions, 2)

class TestEvictionScenarios(unittest.TestCase):
    def test_sequential_access(self):
        c = LRUCache(capacity=3)
        for i in range(1, 4):
            c.put(f"p{i}", {"name": f"Product {i}"})
        for i in range(1, 4):
            c.get(f"p{i}")
        c.put("p4", {"name": "Product 4"})
        self.assertIsNone(c.get("p1"))
        self.assertIsNotNone(c.get("p4"))
    def test_repeated_access(self):
        c = LRUCache(capacity=3)
        for i in range(1, 4):
            c.put(f"p{i}", {"name": f"Product {i}"})
        for _ in range(5):
            c.get("p1")
        c.put("p4", {"name": "Product 4"})
        self.assertIsNone(c.get("p2"))
        self.assertIsNotNone(c.get("p1"))
    def test_update_eviction(self):
        c = LRUCache(capacity=3)
        for i in range(1, 4):
            c.put(f"p{i}", {"price": i * 10})
        c.put("p1", {"price": 15})
        c.put("p4", {"price": 40})
        self.assertIsNone(c.get("p2"))
        self.assertIsNotNone(c.get("p1"))

class TestProductPageCache(unittest.TestCase):
    def setUp(self):
        self.cache = ProductPageCache(capacity=3)
    def test_cache_product(self):
        self.cache.cache_product("p1", {"title": "Laptop", "price": 999.99})
        p = self.cache.get_product("p1")
        self.assertEqual(p["title"], "Laptop")
        self.assertEqual(p["price"], 999.99)
    def test_stats(self):
        self.cache.cache_product("p1", {"title": "P1"})
        self.cache.get_product("p1")
        self.cache.get_product("p2")
        s = self.cache.get_stats()
        self.assertEqual(s["hits"], 1)
        self.assertEqual(s["misses"], 1)
    def test_product_eviction(self):
        for i in range(1, 4):
            self.cache.cache_product(f"p{i}", {"title": f"Product {i}"})
        self.cache.get_product("p1")
        self.cache.cache_product("p4", {"title": "Product 4"})
        self.assertIsNone(self.cache.get_product("p2"))

class TestEdgeCases(unittest.TestCase):
    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            LRUCache(capacity=0)
        with self.assertRaises(ValueError):
            LRUCache(capacity=-1)
    def test_single_capacity(self):
        c = LRUCache(capacity=1)
        c.put("k1", "v1")
        c.put("k2", "v2")
        self.assertIsNone(c.get("k1"))
        self.assertEqual(c.get("k2"), "v2")

def run_performance_test():
    print("\n" + "="*50)
    print("Performance Test")
    print("="*50)
    cache = ProductPageCache(capacity=100)
    for i in range(200):
        cache.cache_product(f"prod{i}", {"title": f"Product {i}", "price": i * 10.0})
    for i in range(150, 200):
        cache.get_product(f"prod{i}")
    for i in range(200, 250):
        cache.get_product(f"prod{i}")
    stats = cache.get_stats()
    print("\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == "test"):
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            sys.argv = [sys.argv[0]]
        print("Running LRU Cache Tests...")
        print("="*50)
        unittest.main(verbosity=2, exit=False)
        run_performance_test()
    else:
        cache = ProductPageCache(capacity=3)
        cache.cache_product("prod1", {"title": "Laptop", "price": 999.99})
        cache.cache_product("prod2", {"title": "Mouse", "price": 29.99})
        cache.cache_product("prod3", {"title": "Keyboard", "price": 79.99})
        print("Example Usage:")
        print(f"Retrieved product: {cache.get_product('prod1')}")
        cache.cache_product("prod4", {"title": "Monitor", "price": 299.99})
        print("\nCache Statistics:")
        for k, v in cache.get_stats().items():
            print(f"  {k}: {v}")
