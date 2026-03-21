#!/usr/bin/env python3
import sys
sys.argv = ['locomo_adapter.py', '--eval', '--api-url', 'http://localhost:5000', '--api-key', 'benchmark_key_locomo_2026', '--granularity', 'turn']
exec(open('/app/benchmark/locomo_adapter.py').read())