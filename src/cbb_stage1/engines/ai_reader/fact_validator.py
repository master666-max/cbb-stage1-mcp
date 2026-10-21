"""Post-validation and semantic cleaning for ChapterFact.

Ported from B06-AI-Reader-V2 ``extraction/fact_validator.py`` (≈1k lines of the
"10 classes of semantic cleaning" referenced by design/00 §4.8 and acceptance
A7).  Changes: imports point at the local schema, and the homonym helper lives
in :mod:`cbb_stage1.engines.ai_reader.location_names` — no ``src.*`` imports.

Location filtering uses the three-layer approach based on Chinese place-name
morphology (专名 + 通名).
"""

from __future__ import annotations

import logging

from cbb_stage1.engines.ai_reader.location_names import is_homonym_prone
from cbb_stage1.engines.ai_reader.schema import (
    ChapterFact,
    CharacterFact,
    EventFact,
    ItemEventFact,
    LocationFact,
    OrgEventFact,
    SpatialRelationship,
    WorldDeclaration,
)

logger = logging.getLogger(__name__)

_VALID_ITEM_ACTIONS = {"出现", "获得", "使用", "赠予", "消耗", "丢失", "损毁"}
_VALID_ORG_ACTIONS = {"加入", "离开", "晋升", "阵亡", "叛出", "逐出"}
_VALID_EVENT_TYPES = {"战斗", "成长", "社交", "旅行", "其他"}
_VALID_IMPORTANCE = {"high", "medium", "low"}
_VALID_SPATIAL_RELATION_TYPES = {
    "direction", "distance", "contains", "adjacent", "separated_by", "terrain",
    "in_between", "travel_path", "relative_scale", "cluster",
}
_VALID_CONFIDENCE = {"high", "medium", "low"}
_VALID_DISTANCE_CLASS = {"near", "medium", "far", "very_far"}

# ── Location name normalization (variant → canonical) ────────────────
_LOCATION_NAME_NORMALIZE: dict[str, str] = {
    "南瞻部洲": "南赡部洲",
    "南赡养部洲": "南赡部洲",
    "南瞻养部洲": "南赡部洲",
}

_NAME_MIN_LEN = 1
_NAME_MIN_LEN_OTHER = 2
_NAME_MAX_LEN = 20

# ── Location morphological validation ─────────────────────────────────

_GEO_GENERIC_SUFFIXES = frozenset(
    "山峰岭崖谷坡"  # mountain
    "河江湖海溪泉潭洋"  # water
    "林森丛"  # forest
    "城楼殿宫庙寺塔洞关门桥台阁堂院府庄园"  # built structures
    "村镇县省国邦州"  # administrative
    "界域洲宗派教"  # fantasy
    "原地坪滩沙漠岛"  # terrain
    "路街道"  # roads
    "屋房舍"  # buildings
)

_POSITIONAL_SUFFIXES = frozenset("上下里内外中前后边旁畔口头脚顶")

_GENERIC_MODIFIERS = frozenset({
    "小", "大", "老", "新", "旧", "那", "这", "某", "一个", "一座", "一片",
    "一条", "一处", "那个", "这个", "那座", "这座",
    "某条", "某个", "某座", "某处", "某片",
})

_CONCEPTUAL_GEO_WORDS = frozenset({
    "江湖", "天下", "世界", "人间", "凡间", "尘世", "世间",
    "世俗界", "修仙界", "仙界", "魔界",
    "地球", "全球", "全世界",
    "中国大陆", "中国", "大陆",
    "外国", "国外", "海外", "世界各地",
})

_VEHICLE_WORDS = frozenset({
    "小舟", "大船", "船只", "马车", "轿子", "飞剑", "法宝",
    "车厢", "船舱", "轿内",
})

_FURNITURE_OBJECT_NAMES = frozenset({
    "炕", "炕上", "炕桌", "板床", "板床上", "榻上", "床上",
    "桌上", "桌下", "书桌", "书案", "案上",
    "椅上", "凳上", "柜中", "柜内", "箱中", "箱内",
    "抽屉", "抽屉内", "小匣", "匣内",
    "火盆", "炉内", "灯下", "烛下",
    "屏风", "帘子", "帘内", "帘外",
    "镜壁", "镜前",
    "门槛", "窗下", "窗前", "窗外",
    "石碣", "碑前",
    "墙角", "墙根",
    "台阶", "阶上",
})

_GENERIC_FACILITY_NAMES = frozenset({
    "酒店", "客店", "客栈", "旅店", "饭店", "酒楼", "酒馆", "酒肆",
    "茶坊", "茶馆", "茶楼", "茶肆", "茶铺",
    "店铺", "铺子", "当铺", "药铺", "药店", "米铺", "布店",
    "集市", "市场", "市集", "庙会",
    "衙门", "公堂", "大堂", "牢房", "牢城", "监牢", "死牢",
    "法场", "刑场", "校场",
    "寺庙", "道观", "庵堂", "祠堂",
    "后堂", "前厅", "正厅", "大厅", "中堂", "花厅",
    "书房", "卧房", "卧室", "厨房", "柴房", "仓库",
    "内室", "内房", "内堂", "后房", "后院", "前院",
    "偏厅", "偏房", "厢房", "耳房",
    "马厩", "马棚", "草料场",
    "山寨", "营寨", "大寨", "寨子",
    "码头", "渡口", "津渡",
    "驿站", "驿馆",
})

_FALLBACK_GEO_BLOCKLIST = frozenset({
    "外面", "里面", "前方", "后方", "旁边", "附近", "远处", "近处",
    "对面", "身边", "身旁", "眼前", "面前", "脚下", "头顶", "上方", "下方",
    "半山腰", "水面", "地面", "天空", "空中",
    "家里", "家中", "家门", "家内",
    "这边", "那边", "这里", "那里", "此地", "此处", "彼处",
    "厅上", "厅前", "厅下", "堂上", "堂前", "堂下",
    "门前", "门外", "门口", "门内", "门下",
    "阶下", "阶前", "廊下", "檐下", "墙外", "墙内",
    "屏风后", "帘后", "帘内",
    "桥头", "桥上", "桥下", "路口", "路上", "路旁", "路边",
    "岸上", "岸边", "水边", "河边", "湖边", "溪边",
    "山上", "山下", "山前", "山后", "山中", "山脚", "山脚下",
    "林中", "林内", "树下", "树林", "草丛",
    "城内", "城外", "城中", "城上", "城下", "城头",
    "村口", "村外", "村中", "村里", "镇上",
    "庄上", "庄前", "庄后", "庄内", "庄外",
    "寨内", "寨外", "寨前", "寨中",
    "店中", "店内", "店外", "店里",
    "房中", "房内", "房里", "屋里", "屋内", "屋中",
    "楼上", "楼下", "楼中",
    "院中", "院内", "院外", "院子",
    "园中", "园内",
    "船上", "船头", "船中",
    "马上", "车上",
    "战场", "阵前", "阵中", "阵后",
    "半路", "途中", "路途", "沿途",
    "深山", "深山老林", "荒山野岭", "穷山恶水",
    "密林深处", "荒野", "旷野", "原野", "野外",
    "树上", "石上", "岩上", "岩石上", "岩石边", "岩石下",
    "石壁", "崖壁", "绝壁",
    "偏僻地方", "偏僻之地", "偏僻之处",
    "神秘之处", "神秘地方", "秘密之处", "隐秘之处",
    "安全之处", "安全地方", "隐蔽之处",
})

# ── Person generic references ─────────────────────────────────────────

_GENERIC_PERSON_WORDS = frozenset({
    "众人", "其他人", "旁人", "来人", "对方", "大家", "所有人",
    "那人", "此人", "其人", "何人", "某人", "外人", "路人",
    "他们", "她们", "我们", "诸位", "各位", "在场众人",
    "妇人", "女子", "汉子", "大汉", "壮汉", "好汉",
    "老儿", "老者", "老翁", "少女", "丫头",
    "军士", "军汉", "兵丁", "喽啰", "小喽啰",
    "差人", "差役", "官差", "公差", "衙役",
    "和尚", "僧人", "道士", "先生", "秀才",
    "店家", "店主", "小二", "店小二", "酒保",
    "庄客", "农夫", "猎户", "渔夫", "樵夫",
    "使者", "信使", "探子", "细作",
    "客人", "客官", "过客", "行人",
})

_PURE_TITLE_WORDS = frozenset({
    "堂主", "长老", "弟子", "护法", "掌门", "帮主", "教主",
    "师父", "师兄", "师弟", "师姐", "师妹", "师傅",
    "大哥", "二哥", "三哥", "大姐", "二姐",
    "侍卫", "仆人", "丫鬟", "小厮",
    "太尉", "知府", "知县", "提辖", "都监", "教头", "都头",
    "将军", "元帅", "丞相", "太师",
    "头领", "寨主", "大王", "员外",
    "恩相", "大人", "老爷", "相公",
})


def _is_generic_location(name: str) -> str | None:
    """Return a filter reason for generic/invalid place names, else None."""
    n = len(name)

    # Rule 1: single-char generic suffix alone.
    if n == 1 and name in _GEO_GENERIC_SUFFIXES:
        return "single-char generic suffix"

    # Rule 2/3/17/4/4b: explicit blocklists.
    if name in _CONCEPTUAL_GEO_WORDS:
        return "conceptual geo word"
    if name in _VEHICLE_WORDS:
        return "vehicle/object"
    if name in _FURNITURE_OBJECT_NAMES:
        return "furniture/object"
    if name in _GENERIC_FACILITY_NAMES:
        return "generic facility name"
    if name in _FALLBACK_GEO_BLOCKLIST:
        return "fallback blocklist"

    # Rule 5: descriptive phrase.
    if "的" in name:
        return "descriptive phrase (contains 的)"
    # Rule 6: too long.
    if n > 7:
        return "too long for a place name"

    # Rule 7: [generic]+[positional suffix].
    if n >= 2 and name[-1] in _POSITIONAL_SUFFIXES:
        prefix = name[:-1]
        if all(c in _GEO_GENERIC_SUFFIXES or c in "场水地天石岩土沙草木树竹" for c in prefix):
            return f"relative position ({prefix}+{name[-1]})"

    # Rule 8: generic modifier + generic suffix.
    if n >= 2:
        for mod in _GENERIC_MODIFIERS:
            if name.startswith(mod):
                rest = name[len(mod):]
                rest_clean = rest.rstrip("子儿")
                if rest_clean and all(c in _GEO_GENERIC_SUFFIXES for c in rest_clean):
                    return f"generic modifier + suffix ({mod}+{rest})"
                break

    # Rule 9: 2-char with both chars generic (but X+州/城/镇/县/国 are real names).
    if n == 2:
        if name[1] not in "州城镇县国省郡府":
            if name[0] in _GEO_GENERIC_SUFFIXES | frozenset("水天地场石土半荒深远近") and name[1] in _GEO_GENERIC_SUFFIXES | frozenset("面子落处口边旁"):
                return "two-char generic compound"

    # Rule 10: demonstrative + directional.
    if n >= 3 and name[-1] in "边里面处" and name[-2] in "这那":
        return "demonstrative + directional"

    # Rule 11: person + location suffix.
    for suf in ("家里", "家中", "那里", "这里", "府上", "住处", "门前", "屋里"):
        if n > len(suf) and name.endswith(suf):
            return f"person + location suffix ({suf})"

    # Rule 12: single-char building part.
    if n == 1 and name in "厅堂楼阁殿亭阶廊柜":
        return "single-char building part"

    # Rule 13: facility + positional.
    if n == 2 and name[1] in "里中内外上下" and name[0] in "店牢庙帐棚洞窑库坑井":
        return "facility + positional"

    # Rule 14: compound positional phrase.
    if n >= 3 and name[-1] in "里中内外上下前后边旁处":
        base = name[:-1]
        _GENERIC_BASES = frozenset({
            "后花园", "前花园", "后院子", "前院子", "大门", "后门", "前门", "侧门",
            "冈子", "山坡", "岭上", "坡下", "崖下", "岸边", "河畔",
            "书案", "桌案", "床头", "窗前", "屏风", "帐帘", "阶梯",
            "墙角", "墙根", "门槛", "门洞", "门扇", "院墙",
        })
        if base in _GENERIC_BASES:
            return f"compound positional ({base}+{name[-1]})"

    # Rule 15: quantifier prefix + filler + generic suffix.
    _QUANT_PREFIXES = (
        "某条", "某个", "某座", "某片", "某处",
        "一条", "一个", "一座", "一片", "一处",
    )
    _GENERIC_TRAIL = (
        "洞穴", "通道", "山洞", "小路", "大路", "小道", "大道",
        "山路", "水路", "地方", "之地", "之处", "峡谷", "山谷",
        "路", "道", "洞", "房", "殿", "厅", "屋", "廊",
    )
    if n >= 3:
        for prefix in _QUANT_PREFIXES:
            if name.startswith(prefix):
                for trail in _GENERIC_TRAIL:
                    if name.endswith(trail) and len(name) <= len(prefix) + 4 + len(trail):
                        return f"quantifier + filler + generic ({prefix}...{trail})"
                break

    # Rule 16: descriptive adjective + generic tail.
    _DESCRIPTIVE_ADJECTIVES = frozenset({
        "偏僻", "荒凉", "偏远", "僻静", "幽静", "隐秘", "神秘",
        "安静", "清幽", "阴暗", "黑暗", "宽敞", "狭窄",
        "破旧", "简陋", "豪华", "广阔",
    })
    _GENERIC_TAILS = frozenset({
        "地方", "之地", "之处", "角落", "所在", "地带", "地界",
    })
    for adj in _DESCRIPTIVE_ADJECTIVES:
        if name.startswith(adj):
            tail = name[len(adj):]
            if tail in _GENERIC_TAILS:
                return f"descriptive adj + generic tail ({adj}+{tail})"
            break

    # Rule 18: character name + room suffix.
    _ROOM_ENDINGS = ("屋内", "屋里", "屋中", "房内", "房里", "房中", "室中", "室内", "室里")
    if n >= 4 and any(name.endswith(e) for e in _ROOM_ENDINGS):
        return "character room suffix"

    return None


_SUFFIX_TO_TYPE: list[tuple[str, str]] = [
    ("大陆", "大陆"), ("山脉", "山脉"), ("山谷", "山谷"),
    ("王国", "王国"), ("帝国", "帝国"),
    ("国", "国"), ("省", "省"), ("州", "州"), ("府", "府"),
    ("城", "城市"), ("镇", "城镇"), ("县", "县"),
    ("村", "村庄"), ("庄", "庄园"),
    ("山", "山"), ("岭", "山岭"), ("峰", "山峰"),
    ("谷", "山谷"), ("崖", "山崖"),
    ("河", "河流"), ("江", "江"), ("湖", "湖泊"),
    ("海", "海"), ("溪", "溪流"), ("泉", "泉"),
    ("洲", "大洲"), ("界", "界域"), ("域", "域"),
    ("洞", "洞府"), ("殿", "宫殿"), ("宫", "宫殿"),
    ("阁", "阁楼"), ("楼", "楼阁"), ("塔", "塔"),
    ("寺", "寺庙"), ("庙", "寺庙"), ("观", "道观"),
    ("岛", "岛屿"), ("关", "关隘"),
    ("门", "门派"), ("宗", "宗门"), ("派", "门派"),
    ("林", "林地"), ("原", "平原"), ("漠", "沙漠"),
    ("园", "园林"), ("轩", "建筑"), ("斋", "建筑"),
]


def _infer_type_from_name(name: str) -> str:
    """Infer location type from the Chinese name suffix (fallback: 区域)."""
    for suffix, type_label in _SUFFIX_TO_TYPE:
        if name.endswith(suffix) and len(name) > len(suffix):
            return type_label
    return "区域"


def _is_generic_person(name: str) -> str | None:
    """Return a filter reason for generic/invalid person names, else None."""
    if name in _GENERIC_PERSON_WORDS:
        return "generic person reference"
    if name in _PURE_TITLE_WORDS:
        return "pure title without surname"
    return None


def _clamp_name(name: str) -> str:
    """Clean and truncate a name (strip list separators, cap length)."""
    name = name.strip()
    for sep in ("、", "，", "；", ",", ";"):
        if sep in name:
            name = name.split(sep)[0].strip()
            break
    if len(name) > _NAME_MAX_LEN:
        return name[:_NAME_MAX_LEN]
    return name


class FactValidator:
    """Validate and clean a :class:`ChapterFact` instance."""

    def __init__(self) -> None:
        # name_corrections: short_name → full_name (built from the dictionary,
        # e.g. {"愣子": "二愣子"}).
        self._name_corrections: dict[str, str] = {}

    def set_name_corrections(self, corrections: dict[str, str]) -> None:
        """Set truncated_name → full_name corrections from the dictionary."""
        self._name_corrections = dict(corrections)

    def validate(self, fact: ChapterFact) -> ChapterFact:
        """Return a cleaned copy of *fact* (never mutates the input)."""
        characters = self._validate_characters(fact.characters)
        relationships = self._validate_relationships(fact.relationships, characters)
        locations = self._validate_locations(fact.locations, characters)
        spatial_relationships = self._validate_spatial_relationships(
            fact.spatial_relationships, locations
        )
        item_events = self._validate_item_events(fact.item_events)
        org_events = self._validate_org_events(fact.org_events)
        events = self._validate_events(fact.events)
        new_concepts = self._validate_concepts(fact.new_concepts)
        world_declarations = self._validate_world_declarations(fact.world_declarations)

        locations = self._ensure_referenced_locations(locations, world_declarations)
        characters = self._remove_locations_from_characters(characters, locations)
        events = self._fill_event_participants(characters, events)
        events = self._fill_event_locations(locations, events)
        characters = self._ensure_participants_in_characters(characters, events)
        characters = self._ensure_relation_persons_in_characters(characters, relationships)
        locations, characters, events, spatial_relationships = (
            self._disambiguate_homonym_locations(
                locations, characters, events, spatial_relationships
            )
        )

        return ChapterFact(
            chapter_id=fact.chapter_id,
            novel_id=fact.novel_id,
            characters=characters,
            relationships=relationships,
            locations=locations,
            spatial_relationships=spatial_relationships,
            item_events=item_events,
            org_events=org_events,
            events=events,
            new_concepts=new_concepts,
            world_declarations=world_declarations,
        )

    # ── characters ────────────────────────────────────────────────

    def _validate_characters(self, chars: list[CharacterFact]) -> list[CharacterFact]:
        seen: dict[str, CharacterFact] = {}
        for ch in chars:
            name = _clamp_name(ch.name)
            if name in self._name_corrections:
                corrected = self._name_corrections[name]
                logger.debug("Name correction: '%s' → '%s'", name, corrected)
                name = corrected
            if len(name) < _NAME_MIN_LEN:
                continue
            reason = _is_generic_person(name)
            if reason:
                logger.debug("Dropping person '%s': %s", name, reason)
                continue
            if name in seen:
                existing = seen[name]
                seen[name] = CharacterFact(
                    name=name,
                    new_aliases=list(dict.fromkeys(existing.new_aliases + ch.new_aliases)),
                    appearance=existing.appearance or ch.appearance,
                    abilities_gained=existing.abilities_gained + ch.abilities_gained,
                    locations_in_chapter=list(
                        dict.fromkeys(existing.locations_in_chapter + ch.locations_in_chapter)
                    ),
                )
            else:
                seen[name] = ch.model_copy(update={"name": name})

        # Alias-based character merge: if A lists B (also an independent
        # character) as an alias, merge B into A.
        merge_targets: dict[str, str] = {}
        for name, ch in seen.items():
            for alias in ch.new_aliases:
                if alias in seen and alias != name:
                    if alias not in merge_targets and name not in merge_targets:
                        merge_targets[alias] = name

        for target, keeper in merge_targets.items():
            if target not in seen or keeper not in seen:
                continue
            target_ch = seen.pop(target)
            keeper_ch = seen[keeper]
            merged_aliases = list(dict.fromkeys(
                keeper_ch.new_aliases + target_ch.new_aliases + [target]
            ))
            merged_aliases = [a for a in merged_aliases if a != keeper]
            merged_locations = list(dict.fromkeys(
                keeper_ch.locations_in_chapter + target_ch.locations_in_chapter
            ))
            seen[keeper] = CharacterFact(
                name=keeper,
                new_aliases=merged_aliases,
                appearance=keeper_ch.appearance or target_ch.appearance,
                abilities_gained=keeper_ch.abilities_gained + target_ch.abilities_gained,
                locations_in_chapter=merged_locations,
            )
            logger.debug("Merged character '%s' into '%s' via explicit alias link", target, keeper)

        # Second pass: clean aliases against the full character set.
        all_names = set(seen.keys())
        for name, ch in seen.items():
            cleaned = self._clean_aliases(ch.new_aliases, name, all_names)
            if len(cleaned) != len(ch.new_aliases):
                seen[name] = ch.model_copy(update={"new_aliases": cleaned})

        return list(seen.values())

    def _clean_aliases(
        self,
        aliases: list[str],
        owner_name: str,
        all_char_names: set[str],
    ) -> list[str]:
        """Drop aliases that are independent characters, too long, or contaminated."""
        cleaned = []
        for alias in aliases:
            if not alias:
                continue
            if alias in all_char_names and alias != owner_name:
                logger.debug(
                    "Alias conflict: '%s' is independent char, removing from %s",
                    alias, owner_name,
                )
                continue
            if len(alias) > 6:
                logger.debug("Alias too long (%d): '%s' for %s", len(alias), alias, owner_name)
                continue
            contaminated = False
            for other in all_char_names:
                if (
                    other != owner_name
                    and len(other) >= 2
                    and other in alias
                    and alias != other
                ):
                    logger.debug(
                        "Alias contains other char: '%s' contains '%s', removing from %s",
                        alias, other, owner_name,
                    )
                    contaminated = True
                    break
            if contaminated:
                continue
            cleaned.append(alias)
        return cleaned

    # ── relationships ─────────────────────────────────────────────

    def _validate_relationships(self, rels, characters):
        char_names = {ch.name for ch in characters}
        for ch in characters:
            char_names.update(ch.new_aliases)

        valid = []
        for rel in rels:
            a = _clamp_name(rel.person_a)
            b = _clamp_name(rel.person_b)
            if len(a) < _NAME_MIN_LEN or len(b) < _NAME_MIN_LEN:
                continue
            if a not in char_names or b not in char_names:
                logger.debug("Dropping relationship %s-%s: person not in characters", a, b)
                continue
            valid.append(rel.model_copy(update={"person_a": a, "person_b": b}))
        return valid

    # ── locations ─────────────────────────────────────────────────

    def _validate_locations(self, locs, characters=None):
        # Split compound names joined by conjunctions: "新房与西院".
        expanded_locs = []
        for loc in locs:
            split_parts = None
            for conj in ("与", "和", "及"):
                if conj in loc.name:
                    idx = loc.name.index(conj)
                    left = loc.name[:idx].strip()
                    right = loc.name[idx + 1:].strip()
                    if len(left) >= 2 and len(right) >= 2:
                        split_parts = [left, right]
                        break
            if split_parts:
                logger.debug("Splitting compound location: '%s' → %s", loc.name, split_parts)
                for part in split_parts:
                    expanded_locs.append(loc.model_copy(update={"name": part}))
            else:
                expanded_locs.append(loc)
        locs = expanded_locs

        char_names: set[str] = set()
        if characters:
            for ch in characters:
                char_names.add(ch.name)
                char_names.update(ch.new_aliases)

        _HALLUCINATED_SUFFIXES = ("府邸", "住所", "居所", "家中", "宅邸", "房间")

        valid = []
        seen_names: set[str] = set()
        for loc in locs:
            name = _clamp_name(loc.name)
            name = _LOCATION_NAME_NORMALIZE.get(name, name)
            if len(name) < _NAME_MIN_LEN_OTHER:
                continue
            if name in seen_names:
                continue
            seen_names.add(name)
            reason = _is_generic_location(name)
            if reason:
                logger.debug("Dropping location '%s': %s", name, reason)
                continue
            if char_names:
                is_hallucinated = False
                for suffix in _HALLUCINATED_SUFFIXES:
                    if name.endswith(suffix):
                        prefix = name[: -len(suffix)]
                        if prefix in char_names:
                            logger.debug(
                                "Dropping hallucinated location: %s (char=%s + suffix=%s)",
                                name, prefix, suffix,
                            )
                            is_hallucinated = True
                            break
                if is_hallucinated:
                    continue
            valid.append(loc.model_copy(update={"name": name}))

        cleaned_valid = []
        for loc in valid:
            if loc.peers:
                valid_peers = [
                    p for p in loc.peers
                    if p and p != loc.name and not _is_generic_location(p)
                ]
                cleaned_valid.append(
                    loc.model_copy(update={"peers": valid_peers if valid_peers else None})
                )
            else:
                cleaned_valid.append(loc)
        return cleaned_valid

    # ── spatial ───────────────────────────────────────────────────

    def _validate_spatial_relationships(
        self, rels: list[SpatialRelationship], locations: list
    ) -> list[SpatialRelationship]:
        loc_names = {loc.name for loc in locations}
        valid = []
        seen: set[tuple[str, str, str]] = set()
        for rel in rels:
            source = _clamp_name(rel.source)
            target = _clamp_name(rel.target)
            source = _LOCATION_NAME_NORMALIZE.get(source, source)
            target = _LOCATION_NAME_NORMALIZE.get(target, target)
            if len(source) < _NAME_MIN_LEN or len(target) < _NAME_MIN_LEN:
                continue
            if source == target:
                continue
            relation_type = rel.relation_type
            if relation_type not in _VALID_SPATIAL_RELATION_TYPES:
                logger.debug("Dropping spatial rel with invalid type: %s", relation_type)
                continue
            confidence = rel.confidence if rel.confidence in _VALID_CONFIDENCE else "medium"
            key = (source, target, relation_type)
            if key in seen:
                continue
            seen.add(key)
            if source not in loc_names and target not in loc_names:
                logger.debug(
                    "Spatial rel %s->%s: neither in current chapter locations", source, target
                )
            evidence = rel.narrative_evidence[:50] if rel.narrative_evidence else ""
            distance_class = rel.distance_class
            if distance_class and distance_class not in _VALID_DISTANCE_CLASS:
                distance_class = None
            confidence_score = rel.confidence_score
            if confidence_score is not None:
                confidence_score = max(0.0, min(1.0, confidence_score))
            valid.append(SpatialRelationship(
                source=source,
                target=target,
                relation_type=relation_type,
                value=rel.value,
                confidence=confidence,
                narrative_evidence=evidence,
                distance_class=distance_class,
                confidence_score=confidence_score,
                waypoints=rel.waypoints,
            ))
        return valid

    # ── items / orgs / events / concepts / declarations ──────────

    def _validate_item_events(self, items: list[ItemEventFact]) -> list[ItemEventFact]:
        valid = []
        for item in items:
            name = _clamp_name(item.item_name)
            if len(name) < _NAME_MIN_LEN_OTHER:
                continue
            action = item.action if item.action in _VALID_ITEM_ACTIONS else "出现"
            valid.append(item.model_copy(update={"item_name": name, "action": action}))
        return valid

    def _validate_org_events(self, orgs: list[OrgEventFact]) -> list[OrgEventFact]:
        valid = []
        for org in orgs:
            name = _clamp_name(org.org_name)
            if len(name) < _NAME_MIN_LEN_OTHER:
                continue
            action = org.action if org.action in _VALID_ORG_ACTIONS else "加入"
            valid.append(org.model_copy(update={"org_name": name, "action": action}))
        return valid

    def _validate_events(self, events: list[EventFact]) -> list[EventFact]:
        valid = []
        seen_summaries: set[str] = set()
        for ev in events:
            if not ev.summary or not ev.summary.strip():
                continue
            summary_key = ev.summary.strip()
            if summary_key in seen_summaries:
                logger.debug("Dropping duplicate event: %s", summary_key[:50])
                continue
            seen_summaries.add(summary_key)
            etype = ev.type if ev.type in _VALID_EVENT_TYPES else "其他"
            importance = ev.importance if ev.importance in _VALID_IMPORTANCE else "medium"
            valid.append(ev.model_copy(update={"type": etype, "importance": importance}))
        return valid

    def _validate_concepts(self, concepts):
        valid = []
        for c in concepts:
            name = _clamp_name(c.name)
            if len(name) < _NAME_MIN_LEN_OTHER:
                continue
            valid.append(c.model_copy(update={"name": name}))
        return valid

    def _validate_world_declarations(
        self, declarations: list[WorldDeclaration]
    ) -> list[WorldDeclaration]:
        valid_types = {"region_division", "layer_exists", "portal", "region_position"}
        valid = []
        for decl in declarations:
            if decl.declaration_type not in valid_types:
                logger.debug(
                    "Dropping world declaration with invalid type: %s", decl.declaration_type
                )
                continue
            if not isinstance(decl.content, dict) or not decl.content:
                continue
            confidence = decl.confidence if decl.confidence in _VALID_CONFIDENCE else "medium"
            evidence = decl.narrative_evidence[:100] if decl.narrative_evidence else ""
            valid.append(WorldDeclaration(
                declaration_type=decl.declaration_type,
                content=decl.content,
                narrative_evidence=evidence,
                confidence=confidence,
            ))
        return valid

    # ── cross-record repairs ─────────────────────────────────────

    def _remove_locations_from_characters(
        self, characters: list[CharacterFact], locations: list
    ) -> list[CharacterFact]:
        loc_names = {loc.name for loc in locations}
        if not loc_names:
            return characters
        cleaned = []
        for ch in characters:
            if ch.name in loc_names:
                logger.debug("Removing location '%s' from characters list", ch.name)
                continue
            cleaned.append(ch)
        return cleaned

    def _fill_event_participants(
        self, characters: list[CharacterFact], events: list[EventFact]
    ) -> list[EventFact]:
        all_names: set[str] = set()
        for ch in characters:
            all_names.add(ch.name)
            all_names.update(ch.new_aliases)
        sorted_names = sorted(all_names, key=len, reverse=True)

        updated = []
        for ev in events:
            if not ev.participants:
                found = []
                for name in sorted_names:
                    if name in ev.summary and name not in found:
                        found.append(name)
                if found:
                    ev = ev.model_copy(update={"participants": found})
            updated.append(ev)
        return updated

    def _fill_event_locations(self, locations: list, events: list[EventFact]) -> list[EventFact]:
        loc_names = sorted([loc.name for loc in locations], key=len, reverse=True)
        updated = []
        for ev in events:
            if not ev.location and loc_names:
                for loc_name in loc_names:
                    if loc_name in ev.summary:
                        ev = ev.model_copy(update={"location": loc_name})
                        break
            updated.append(ev)
        return updated

    def _ensure_participants_in_characters(
        self, characters: list[CharacterFact], events: list[EventFact]
    ) -> list[CharacterFact]:
        char_names = {ch.name for ch in characters}
        for ch in characters:
            char_names.update(ch.new_aliases)
        for ev in events:
            for p in ev.participants:
                p = p.strip()
                if p and p not in char_names and len(p) >= _NAME_MIN_LEN and not _is_generic_person(p):
                    characters.append(CharacterFact(name=p))
                    char_names.add(p)
                    logger.debug("Auto-added character from event participant: %s", p)
        return characters

    def _ensure_relation_persons_in_characters(
        self, characters: list[CharacterFact], relationships
    ) -> list[CharacterFact]:
        char_names = {ch.name for ch in characters}
        for ch in characters:
            char_names.update(ch.new_aliases)
        for rel in relationships:
            for name in (rel.person_a, rel.person_b):
                name = name.strip()
                if name and name not in char_names and len(name) >= _NAME_MIN_LEN and not _is_generic_person(name):
                    characters.append(CharacterFact(name=name))
                    char_names.add(name)
                    logger.debug("Auto-added character from relationship: %s", name)
        return characters

    def _ensure_referenced_locations(
        self,
        locations: list,
        world_declarations: list[WorldDeclaration],
    ) -> list:
        """Auto-create entries for referenced parents / declaration names."""
        existing_names = {loc.name for loc in locations}
        to_add: dict[str, LocationFact] = {}

        for loc in locations:
            parent = loc.parent
            if parent:
                parent = _LOCATION_NAME_NORMALIZE.get(parent.strip(), parent.strip())
            if parent and parent not in existing_names and parent not in to_add:
                to_add[parent] = LocationFact(
                    name=parent,
                    type=_infer_type_from_name(parent),
                    description="",
                )
                logger.debug("Auto-adding parent location: %s (referenced by %s)", parent, loc.name)

        for decl in world_declarations:
            content = decl.content
            if decl.declaration_type == "region_division":
                for child in content.get("children", []):
                    child = child.strip()
                    if child and child not in existing_names and child not in to_add:
                        to_add[child] = LocationFact(
                            name=child,
                            type=_infer_type_from_name(child),
                            parent=content.get("parent"),
                            description="",
                        )
                        logger.debug("Auto-adding location from region_division: %s", child)
                div_parent = content.get("parent", "")
                if div_parent and div_parent.strip():
                    div_parent = div_parent.strip()
                    if div_parent not in existing_names and div_parent not in to_add:
                        to_add[div_parent] = LocationFact(
                            name=div_parent,
                            type=_infer_type_from_name(div_parent),
                            description="",
                        )
                        logger.debug("Auto-adding location from region_division parent: %s", div_parent)
            elif decl.declaration_type == "portal":
                for key in ("source_location", "target_location"):
                    loc_name = content.get(key, "")
                    if loc_name and loc_name.strip():
                        loc_name = loc_name.strip()
                        if loc_name not in existing_names and loc_name not in to_add:
                            to_add[loc_name] = LocationFact(
                                name=loc_name,
                                type="地点",
                                description="",
                            )
                            logger.debug("Auto-adding location from portal: %s", loc_name)

        if to_add:
            locations = locations + list(to_add.values())
            logger.info(
                "Auto-added %d referenced locations: %s",
                len(to_add), ", ".join(to_add.keys()),
            )
        return locations

    def _disambiguate_homonym_locations(
        self,
        locations: list,
        characters: list[CharacterFact],
        events: list[EventFact],
        spatial_relationships: list[SpatialRelationship],
    ) -> tuple[list, list[CharacterFact], list[EventFact], list[SpatialRelationship]]:
        rename_map: dict[str, str] = {}

        new_locations = []
        for loc in locations:
            if is_homonym_prone(loc.name) and loc.parent:
                new_name = f"{loc.parent}·{loc.name}"
                rename_map[loc.name] = new_name
                new_locations.append(loc.model_copy(update={"name": new_name}))
                logger.debug("Disambiguated location: '%s' → '%s'", loc.name, new_name)
            else:
                new_locations.append(loc)

        if not rename_map:
            return locations, characters, events, spatial_relationships

        logger.info(
            "Disambiguated %d homonym locations: %s",
            len(rename_map), ", ".join(f"{k}→{v}" for k, v in rename_map.items()),
        )

        new_characters = []
        for ch in characters:
            new_locs = [rename_map.get(loc, loc) for loc in ch.locations_in_chapter]
            if new_locs != list(ch.locations_in_chapter):
                new_characters.append(ch.model_copy(update={"locations_in_chapter": new_locs}))
            else:
                new_characters.append(ch)

        new_events = []
        for ev in events:
            if ev.location and ev.location in rename_map:
                new_events.append(ev.model_copy(update={"location": rename_map[ev.location]}))
            else:
                new_events.append(ev)

        new_spatial = []
        for rel in spatial_relationships:
            updates: dict = {}
            if rel.source in rename_map:
                updates["source"] = rename_map[rel.source]
            if rel.target in rename_map:
                updates["target"] = rename_map[rel.target]
            new_spatial.append(rel.model_copy(update=updates) if updates else rel)

        final_locations = []
        for loc in new_locations:
            if loc.parent and loc.parent in rename_map:
                final_locations.append(loc.model_copy(update={"parent": rename_map[loc.parent]}))
            else:
                final_locations.append(loc)

        return final_locations, new_characters, new_events, new_spatial
