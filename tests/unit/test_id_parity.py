from __future__ import annotations

from cbb_stage1 import ids
from cbb_stage1.box.records import make_entity_id
from cbb_stage1.cues.state import cue_id_of
from cbb_stage1.resolve.merges import merge_id_for


def test_entity_merge_cue_wrappers_delegate_to_ids():
    assert make_entity_id("zj", "person", "贝塔心叶") == ids.entity_id(
        "zj", "person", "贝塔心叶"
    )
    assert merge_id_for("zj", "alias", ["e:2", "e:1"], "e:3") == ids.merge_id(
        "zj", "alias", ["e:2", "e:1"], "e:3"
    )
    cue = {
        "book_id": "zj",
        "kind": "object",
        "subject": {"literal": "影子"},
        "setup": {"unit_id": None},
    }
    assert cue_id_of(cue) == ids.cue_id("zj", "object", {"literal": "影子"}, None)
