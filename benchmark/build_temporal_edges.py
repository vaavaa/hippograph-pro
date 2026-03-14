#!/usr/bin/env python3
"""
Variant C: Build TEMPORAL_BEFORE / TEMPORAL_AFTER edges.

Работает через API контейнера. Не трогает продакшн.
Запускать на hippograph-temporal (порт 5005).

Протокол:
- Берёт все ноды с t_event_start из БД
- Сортирует по t_event_start
- Для каждой ноды создаёт рёбра к ±3 ближайшим соседям
- TEMPORAL_BEFORE: старая нота -> новая нота
- TEMPORAL_AFTER:  новая нота -> старая нота
- weight: 0.4

Checkpoint: benchmark/results/temporal_edges_progress.json
"""

import sqlite3
import json
import os
import sys
from datetime import datetime

DB_PATH = os.environ.get('DB_PATH', '/app/data/temporal_test.db')
TEMPORAL_WEIGHT = 0.4
NEIGHBORS = 3  # ±3 ближайших по времени
PROGRESS_FILE = '/app/benchmark/results/temporal_edges_progress.json'


def get_connection(db_path):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def get_existing_temporal_edges(cursor):
    """Загрузить уже существующие temporal рёбра (для идемпотентности)."""
    cursor.execute(
        "SELECT source_id, target_id FROM edges WHERE edge_type IN ('TEMPORAL_BEFORE', 'TEMPORAL_AFTER')"
    )
    return set((r['source_id'], r['target_id']) for r in cursor.fetchall())


def create_edge(cursor, source_id, target_id, edge_type, weight, existing):
    """Создать ребро если его ещё нет."""
    key = (source_id, target_id)
    if key in existing:
        return False
    cursor.execute(
        "INSERT OR IGNORE INTO edges (source_id, target_id, weight, edge_type) VALUES (?, ?, ?, ?)",
        (source_id, target_id, weight, edge_type)
    )
    existing.add(key)
    return True


def main():
    print('=' * 60)
    print('VARIANT C: Build Temporal Edges')
    print(f'DB: {DB_PATH}')
    print(f'Neighbors: ±{NEIGHBORS}')
    print(f'Weight: {TEMPORAL_WEIGHT}')
    print('=' * 60)

    conn = get_connection(DB_PATH)
    cursor = conn.cursor()

    # 1. Загрузить все ноды с t_event_start
    cursor.execute(
        "SELECT id, t_event_start FROM nodes "
        "WHERE t_event_start IS NOT NULL "
        "ORDER BY t_event_start ASC"
    )
    nodes = [(r['id'], r['t_event_start']) for r in cursor.fetchall()]
    print(f'\nНоды с t_event_start: {len(nodes)}')

    if len(nodes) < 2:
        print('Недостаточно нод с t_event_start для построения рёбер.')
        conn.close()
        return

    # 2. Загрузить существующие temporal рёбра
    existing = get_existing_temporal_edges(cursor)
    print(f'Уже существующих temporal рёбер: {len(existing)}')

    # 3. Построить рёбра
    created_before = 0
    created_after = 0
    skipped = 0

    for i, (node_id, t_event) in enumerate(nodes):
        # Соседи слева (раньше по времени) — создаём TEMPORAL_BEFORE
        left_start = max(0, i - NEIGHBORS)
        for j in range(left_start, i):
            older_id = nodes[j][0]
            # older -> current = TEMPORAL_BEFORE (старая ведёт к новой)
            if create_edge(cursor, older_id, node_id, 'TEMPORAL_BEFORE', TEMPORAL_WEIGHT, existing):
                created_before += 1
            # current -> older = TEMPORAL_AFTER (новая ссылается на старую)
            if create_edge(cursor, node_id, older_id, 'TEMPORAL_AFTER', TEMPORAL_WEIGHT, existing):
                created_after += 1

        if i % 500 == 0 and i > 0:
            conn.commit()
            print(f'  Progress: {i}/{len(nodes)} нод, '
                  f'BEFORE={created_before}, AFTER={created_after}')

    conn.commit()

    # 4. Итог
    total_created = created_before + created_after
    print(f'\n{"=" * 60}')
    print(f'ГОТОВО')
    print(f'  TEMPORAL_BEFORE рёбер создано: {created_before}')
    print(f'  TEMPORAL_AFTER  рёбер создано: {created_after}')
    print(f'  Итого новых рёбер:             {total_created}')
    print(f'  Пропущено (уже существовали): {skipped}')

    # 5. Checkpoint
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    result = {
        'step': 'temporal_edges_built',
        'timestamp': datetime.now().isoformat(),
        'nodes_with_t_event': len(nodes),
        'temporal_before_created': created_before,
        'temporal_after_created': created_after,
        'total_created': total_created,
        'db_path': DB_PATH
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nCheckpoint: {PROGRESS_FILE}')
    print('Следующий шаг: запустить run_variant_c.py')

    conn.close()


if __name__ == '__main__':
    main()