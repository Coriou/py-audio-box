"""Maintenance accepts the immutable catalogue without admitting foreign SQL."""
import sqlite3

import pytest

from service.recovery import _maintenance_connection, _maintenance_schema_objects
from service.voice_store import _TABLE_DDL, _TRIGGER_DDL


@pytest.mark.parametrize('change', ['exact', 'body', 'extra-trigger', 'index', 'view', 'partial'])
def test_catalogue_maintenance_schema(change):
    with sqlite3.connect(':memory:') as db:
        for sql in (*_TABLE_DDL, *_TRIGGER_DDL):
            db.execute(sql)
        if change == 'body':
            db.execute('DROP TRIGGER catalogue_versions_no_update')
            db.execute("CREATE TRIGGER catalogue_versions_no_update BEFORE UPDATE ON catalogue_versions BEGIN SELECT 1; END")
        elif change == 'extra-trigger':
            db.execute('CREATE TRIGGER foreign_trigger BEFORE INSERT ON catalogue_versions BEGIN SELECT 1; END')
        elif change == 'index':
            db.execute('CREATE INDEX foreign_index ON catalogue_versions(id)')
        elif change == 'view':
            db.execute('CREATE VIEW foreign_view AS SELECT * FROM catalogue_versions')
        elif change == 'partial':
            db.execute('DROP TABLE catalogue_requests')
        if change == 'exact':
            _maintenance_schema_objects(db)
            _maintenance_connection(db)
        else:
            with pytest.raises(ValueError, match='b2_unknown_schema'):
                _maintenance_connection(db)


def test_empty_private_copy_connection_is_allowed():
    with sqlite3.connect(':memory:') as db:
        _maintenance_connection(db)
        with pytest.raises(ValueError, match='b2_unknown_schema'):
            _maintenance_schema_objects(db)
