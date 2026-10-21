"""Entity pre-scan: full-text statistical scanning + optional LLM classification.

Ported from B06-AI-Reader-V2 ``extraction/entity_pre_scanner.py`` (Phase 1 +
Phase 2).  Changes:

* chapter loading comes from the caller (ingest owns chapters), not from a DB;
* dictionary persistence goes through :class:`DictionaryStore`;
* ``jieba`` is an **optional** dependency: when it is not installed the scanner
  still runs every regex/ngram/dialogue/naming pass and reports
  ``jieba_available == False`` instead of failing the pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter

from cbb_stage1.engines.ai_reader.prompts import build_prescan_prompt
from cbb_stage1.engines.ai_reader.schema import EntityDictEntry
from cbb_stage1.engines.ai_reader.stores import (
    PRESCAN_COMPLETED,
    PRESCAN_FAILED,
    PRESCAN_RUNNING,
    DictionaryStore,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suffix rules for entity type inference
# ---------------------------------------------------------------------------
_SUFFIX_RULES: dict[str, list[str]] = {
    "location": [
        "山", "洞", "洲", "国", "城", "宫", "殿", "府", "寺", "庙", "观", "院",
        "阁", "楼", "塔", "谷", "崖", "峰", "岭", "河", "海", "湖", "泊", "关",
        "门", "桥", "镇", "村", "庄", "寨", "营", "港", "岛", "坊", "台", "池",
        "林", "园", "原", "坡", "涧",
    ],
    "org": [
        "派", "宗", "帮", "会", "盟", "教", "族", "军", "营", "卫", "堂",
        "门派", "宗门",
    ],
    "item": [
        "丹", "药", "剑", "刀", "枪", "珠", "鼎", "炉", "符", "阵", "经", "诀",
        "功", "术", "法", "戟", "斧", "锤", "弓", "甲", "袍", "环", "镯", "钟",
        "琴", "笛", "扇", "杖", "棍", "鞭", "索", "旗", "印", "册", "图", "镜",
    ],
    "person": [
        "真人", "道人", "仙人", "大师", "长老", "掌门", "圣人", "大王",
        "将军", "元帅", "太子", "公主", "娘娘", "老祖", "仙子", "童子",
        "居士", "道长", "法师", "尊者", "菩萨", "罗汉",
    ],
}

# ---------------------------------------------------------------------------
# Stopwords — common non-entity high-frequency words
# ---------------------------------------------------------------------------
_STOPWORDS: set[str] = {
    # Verbs / adverbs
    "然后", "但是", "不过", "因为", "如果", "已经", "虽然", "而且", "或者",
    "所以", "因此", "于是", "不是", "就是", "只是", "可是", "还是", "只有",
    "不能", "不会", "没有", "不要", "可以", "应该", "能够", "需要", "知道",
    "觉得", "认为", "看到", "听到", "感觉", "发现", "开始", "继续", "出来",
    "出去", "进来", "进去", "起来", "下去", "回来", "过来", "过去", "上去",
    "不得", "不了", "下来", "进了", "到了", "去了", "来了", "走了", "在此",
    # Pronouns / determiners
    "什么", "这个", "那个", "这些", "那些", "自己", "他们", "她们", "我们",
    "你们", "大家", "别人", "其他", "所有", "一切", "任何", "每个", "这样",
    "那样", "怎么", "怎样", "如何", "为何", "为什么", "哪里", "哪个",
    "那里", "这里", "何处", "何人", "何事", "此处", "此人", "此事",
    # Measure words / time / common words
    "一个", "两个", "一些", "一下", "一次", "一时", "此时", "当时", "顿时",
    "这时", "那时", "同时", "时候", "时间", "地方", "东西", "事情", "问题",
    "了一", "了个", "着一", "了这", "一面", "一阵", "一身", "一般", "一番",
    # Conjunctions / particles
    "不禁", "只见", "忽然", "突然", "果然", "居然", "竟然", "终于", "马上",
    "立刻", "随即", "正是", "便是", "却是", "原来", "本来", "看来", "说道",
    "一声", "一看", "只道", "心中", "心想", "暗想", "想到", "说完", "心下",
    # Common verbs in classical Chinese
    "不曾", "须是", "却说", "话说", "且说", "只因", "正在", "休得", "休要",
    "如今", "只是", "甚么", "如何", "那厮", "这厮",
    # Common nouns that are NOT entity names
    "众人", "小人", "哥哥", "兄弟", "妇人", "老爷", "丫头", "夫人", "师父",
    "弟子", "师兄", "师弟", "师姐", "师妹", "徒弟", "父亲", "母亲", "兄长",
    "大人", "老人", "小姐", "公子", "先生", "姑娘", "奴婢", "丞相", "使者",
    "一人", "二人", "三人", "四人", "五人", "十人", "百人",
    "好汉", "英雄", "壮士", "头领", "先锋", "军马", "人马", "军兵",
    "官兵", "兵马", "将士", "弟兄", "朋友", "客人", "主人",
    # Common verbs appearing as nouns in jieba
    "听得", "看了", "看得", "说了", "听了", "走了", "来了", "去了",
    "只得", "不敢", "不知", "不见", "不曾", "只管", "只得",
    "见了", "问道", "答道", "回道", "叫道", "今日", "商议",
    "如此", "如此这般", "只好", "便去", "便道", "上前",
    # Directional / positional
    "里面", "外面", "上面", "下面", "前面", "后面", "左右", "身边",
    "面前", "旁边", "其中", "之中", "之上", "之下", "之间",
    "陛下", "殿下", "阁下",
    # Counters / numbers in text
    "一个", "两个", "三个", "四个", "五个", "十个", "几个",
    "个人", "二人", "多人", "十余", "数十", "百余",
    # Common non-entity words in classical Chinese novels
    "官人", "那人", "大喜", "大怒", "上山", "下山", "回来",
    "那妇人", "此人", "这人", "一行", "太公", "军士", "军兵",
    "知府", "管营", "差人", "衙内",
    # Common words that appear before dialogue verbs but aren't names
    "对方", "无法", "原本", "全都", "有点", "立刻", "马上", "似乎",
    "果然", "只好", "最后", "终于", "突然", "忽然", "随后", "接着",
    "竟然", "居然", "顿时", "于是", "赶紧", "连忙", "急忙", "不由",
    "身份", "老者", "少女", "女子", "男子", "大汉", "少年", "青年",
    "仙师", "道友", "前辈", "晚辈", "小子", "丫头", "老头", "老人家",
    "天子", "庄客", "小弟",
    # Dialogue verb prefixes (can be mistaken for speaker names)
    "冷笑", "微笑", "苦笑", "嗤笑", "大笑", "大叫", "大喝", "大怒",
    "消息", "人家", "总算", "差点", "师傅", "中年人", "少妇", "儒生",
    "老道", "的说", "有些", "根本", "当然", "一定", "这才", "那就",
    "这就", "就要", "将要", "正要", "又要", "老夫", "本座", "在下",
    # More dialogue/action false positives
    "命令", "现在", "多谢", "黑脸", "血光", "大哥", "也不", "小心",
    "赶快", "快点", "快些", "随便", "难道", "到底", "究竟", "好像",
    "不过", "只要", "只能", "不如", "还有", "没想到", "想不到",
}

# Patterns to strip from candidates (e.g., "宋江道" -> "宋江")
_DIALOGUE_VERB_SUFFIX = re.compile(
    r'^([\u4e00-\u9fff]{2,6})'
    r'(道|说|曰|笑道|叫道|问道|答道|喝道|叹道|骂道|喊道|怒道|惊道|忙道|急道)$'
)

# ---------------------------------------------------------------------------
# Dialogue attribution patterns
# ---------------------------------------------------------------------------
_DIALOGUE_VERBS = (
    r'(?:道|说|曰|笑道|叫道|问道|答道|喝道|叹道|骂道|喊道|怒道|惊道|忙道|急道'
    r'|冷笑道|大叫道|大喝道|大笑道|冷冷道|淡淡道|沉声道|低声道|高声道'
    r'|嗤笑道|微笑道|苦笑道|轻声道|朗声道|厉声道|柔声道|冷声道|冷哼道)'
)

# Pattern 1: "dialogue" X道 (classical)
_DIALOGUE_PATTERN = re.compile(
    r'[\u201c\u201d""「『]([^""」』\u201c\u201d]{1,200})[\u201c\u201d""」』]\s*[，,。]?\s*'
    r'([\u4e00-\u9fff]{2,6})\s*'
    + _DIALOGUE_VERBS
)

# Pattern 2: X道："dialogue" (modern, speaker before colon+quote)
_SPEAKER_BEFORE_PATTERN = re.compile(
    r'([\u4e00-\u9fff]{2,6})\s*'
    + _DIALOGUE_VERBS
    + r'\s*[：:]\s*[\u201c""「『]'
)

# Pattern 3: X + dialogue verb (modern web novel, no quote needed)
_BARE_SPEAKER_PATTERN = re.compile(
    r'(?:^|[，。！？；：、\n\r""」』])\s*'
    r'([\u4e00-\u9fff]{2,4})'
    + _DIALOGUE_VERBS,
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Naming introduction patterns — "叫作二愣子", "名叫韩立", "绰号行者"
# ---------------------------------------------------------------------------
_NAMING_PATTERN = re.compile(
    r'(?:叫作|叫做|名叫|绰号|外号|人称|又叫|又名|唤作|唤做|自称|名为|号称|名号)'
    r'[\u201c「]?'
    r'(?![的了是在有他她我你它为不])'
    r'([\u4e00-\u9fff]{2,4})'
    r'(?=[\u201d」\s，,。.！!？?、；;：:的了呢吧啊哦\u2018\u2019]|$)'
)


def jieba_available() -> bool:
    """True when the optional ``jieba`` dependency can be imported."""
    try:
        import jieba  # noqa: F401
        import jieba.posseg  # noqa: F401
    except ImportError:
        return False
    return True


class EntityPreScanner:
    """Scans novel text to build the entity dictionary (candidate layer only)."""

    def __init__(
        self,
        dictionary_store: DictionaryStore | None = None,
        llm=None,
        *,
        enable_llm: bool = True,
    ) -> None:
        self.store = dictionary_store
        self.llm = llm
        self.enable_llm = enable_llm and llm is not None
        self.jieba_available = jieba_available()
        if not self.jieba_available:
            logger.warning(
                "jieba is not installed: pre-scan runs ngram/dialogue/naming/suffix "
                "passes only (word-frequency scan skipped)"
            )

    # ── main entry ────────────────────────────────────────────────

    async def scan(
        self,
        book_id: str,
        chapters: list[str],
        titles: list[str] | None = None,
    ) -> list[EntityDictEntry]:
        """Run Phase 1 + Phase 2 and persist through the dictionary store."""
        logger.info("预扫描开始: book_id=%s chapters=%d", book_id, len(chapters))
        if self.store is not None:
            self.store.set_status(book_id, PRESCAN_RUNNING)
        try:
            if not chapters:
                logger.warning("预扫描跳过: 无章节数据 book_id=%s", book_id)
                if self.store is not None:
                    self.store.set_status(book_id, PRESCAN_COMPLETED)
                return []

            titles = list(titles or [])
            full_text = "\n".join(chapters)

            candidates = await asyncio.to_thread(self._phase1_scan, chapters, titles, full_text)
            logger.info("Phase 1 完成: %d 个候选实体, book_id=%s", len(candidates), book_id)

            if self.enable_llm:
                try:
                    candidates = await self._classify_with_llm(candidates)
                    logger.info("Phase 2 完成: %d 个实体, book_id=%s", len(candidates), book_id)
                except Exception:  # noqa: BLE001 - phase 2 is best-effort
                    logger.warning("Phase 2 LLM 分类失败，使用 Phase 1 统计结果", exc_info=True)

            if self.store is not None:
                self.store.replace_entries(book_id, candidates)
                self.store.set_status(book_id, PRESCAN_COMPLETED)
            logger.info("预扫描完成: %d 个实体, book_id=%s", len(candidates), book_id)
            return candidates
        except Exception:
            logger.error("预扫描失败: book_id=%s", book_id, exc_info=True)
            if self.store is not None:
                self.store.set_status(book_id, PRESCAN_FAILED)
            raise

    # ── Phase 1: statistical scan (sync, runs via to_thread) ──────

    def _phase1_scan(
        self,
        chapters: list[str],
        titles: list[str],
        full_text: str,
    ) -> list[EntityDictEntry]:
        word_freq = self._scan_word_freq(full_text)
        ngram_freq = self._scan_ngrams(full_text)
        dialogue_names = self._extract_dialogue_names(full_text)
        title_words = self._extract_title_words(titles)
        naming_names = self._extract_naming_patterns(full_text)

        all_names = (
            set(word_freq.keys()) | set(ngram_freq.keys())
            | set(dialogue_names.keys()) | set(title_words.keys())
            | set(naming_names.keys())
        )
        suffix_types = self._match_suffix_patterns(all_names)

        return self._merge_candidates(
            word_freq, ngram_freq, dialogue_names, title_words,
            naming_names, suffix_types, full_text,
        )

    _NUM_PREFIXES = frozenset("一二三四五六七八九十")
    _NUMERAL_CHARS = frozenset("一二三四五六七八九十百千万亿零两")

    def _scan_word_freq(self, full_text: str) -> Counter:
        """jieba POS-tagged word frequency: keep nouns 2-8 chars (1M char cap)."""
        if not self.jieba_available:
            return Counter()
        import jieba.posseg as pseg

        text = full_text[:1_000_000] if len(full_text) > 1_000_000 else full_text
        counter: Counter = Counter()
        for word, flag in pseg.cut(text):
            if len(word) < 2 or len(word) > 8:
                continue
            if word in _STOPWORDS:
                continue
            if flag.startswith(("nr", "ns", "nz", "n")):
                counter[word] += 1
            elif (
                len(word) >= 3
                and word[0] in self._NUM_PREFIXES
                and not all(c in self._NUMERAL_CHARS for c in word)
            ):
                counter[word] += 1

        if len(full_text) > 1_000_000:
            ratio = len(full_text) / 1_000_000
            for word in counter:
                counter[word] = int(counter[word] * ratio)
        return counter

    def _scan_ngrams(
        self, full_text: str, min_n: int = 2, max_n: int = 4, min_freq: int = 5
    ) -> Counter:
        """Character-level n-gram frequency to catch words jieba missed (500K cap)."""
        text = full_text[:500_000] if len(full_text) > 500_000 else full_text
        segments = re.findall(r'[\u4e00-\u9fff]{2,}', text)

        counter: Counter = Counter()
        for seg in segments:
            for n in range(min_n, min(max_n + 1, len(seg) + 1)):
                for i in range(len(seg) - n + 1):
                    counter[seg[i:i + n]] += 1

        return Counter({
            gram: freq for gram, freq in counter.items()
            if freq >= min_freq and gram not in _STOPWORDS
        })

    def _extract_dialogue_names(self, full_text: str) -> Counter:
        """Extract speaker names from the three dialogue attribution patterns."""
        counter: Counter = Counter()
        for regex, group, max_len in (
            (_DIALOGUE_PATTERN, 2, 6),
            (_SPEAKER_BEFORE_PATTERN, 1, 6),
            (_BARE_SPEAKER_PATTERN, 1, 4),
        ):
            for match in regex.finditer(full_text):
                name = match.group(group).strip()
                if 2 <= len(name) <= max_len and name not in _STOPWORDS:
                    counter[name] += 1
        return counter

    def _extract_title_words(self, titles: list[str]) -> Counter:
        """Extract meaningful words from chapter titles."""
        counter: Counter = Counter()
        for title in titles:
            cleaned = re.sub(r'^第[一二三四五六七八九十百千万零\d]+[章回节卷篇][\s：:]*', '', title)
            cleaned = re.sub(r'^\d+[\s\.、：:]+', '', cleaned)
            if not cleaned:
                continue
            if self.jieba_available:
                import jieba

                words = list(jieba.cut(cleaned))
            else:
                words = re.findall(r'[\u4e00-\u9fff]{2,6}', cleaned)
            for word in words:
                if len(word) < 2 or len(word) > 6:
                    continue
                if word in _STOPWORDS:
                    continue
                if all('\u4e00' <= c <= '\u9fff' for c in word):
                    counter[word] += 1
        return counter

    def _extract_naming_patterns(self, full_text: str) -> Counter:
        """Extract names introduced via explicit naming patterns (2M char cap)."""
        text = full_text[:2_000_000] if len(full_text) > 2_000_000 else full_text
        counter: Counter = Counter()
        for match in _NAMING_PATTERN.finditer(text):
            name = match.group(1).strip()
            if name and name not in _STOPWORDS:
                counter[name] += 1
        return counter

    def _match_suffix_patterns(self, candidates: set[str]) -> dict[str, str]:
        """Match suffix patterns to infer entity types (prefix must be ≥2 chars)."""
        result: dict[str, str] = {}
        for name in candidates:
            if len(name) < 3:
                continue
            for entity_type, suffixes in _SUFFIX_RULES.items():
                for suffix in suffixes:
                    prefix_len = len(name) - len(suffix)
                    if prefix_len < 2:
                        continue
                    if name.endswith(suffix):
                        result[name] = entity_type
                        break
                if name in result:
                    break
        return result

    def _merge_candidates(
        self,
        word_freq: Counter,
        ngram_freq: Counter,
        dialogue_names: Counter,
        title_words: Counter,
        naming_names: Counter,
        suffix_types: dict[str, str],
        full_text: str,
    ) -> list[EntityDictEntry]:
        """Merge all sources into a de-duplicated candidate list."""
        for source in (word_freq, ngram_freq):
            to_redirect: list[tuple[str, str, int]] = []
            for name, freq in list(source.items()):
                match = _DIALOGUE_VERB_SUFFIX.match(name)
                if match:
                    to_redirect.append((name, match.group(1), freq))
            for dirty, clean, freq in to_redirect:
                del source[dirty]
                source[clean] = source.get(clean, 0) + freq

        jieba_names = set(word_freq.keys())
        deduped_ngrams = self._dedup_ngrams(ngram_freq, jieba_names)
        ngram_only = Counter({
            gram: freq for gram, freq in deduped_ngrams.items()
            if gram not in jieba_names
        })

        all_names: set[str] = set()
        all_names.update(word_freq.keys())
        all_names.update(ngram_only.keys())
        all_names.update(dialogue_names.keys())
        all_names.update(title_words.keys())
        all_names.update(naming_names.keys())

        # Numeric-prefix upgrade: "愣子" -> "二愣子" when the long form exists.
        short_to_remove: set[str] = set()
        for name in list(all_names):
            if len(name) >= 3 and name[0] in self._NUM_PREFIXES:
                short_form = name[1:]
                if short_form in all_names and short_form not in short_to_remove:
                    long_freq = max(
                        word_freq.get(name, 0), ngram_only.get(name, 0),
                        dialogue_names.get(name, 0), title_words.get(name, 0),
                        naming_names.get(name, 0),
                    )
                    if long_freq >= 3:
                        short_to_remove.add(short_form)
                        for source in (word_freq, ngram_only, dialogue_names, title_words, naming_names):
                            if short_form in source:
                                source[name] = source.get(name, 0) + source.pop(short_form)
                        logger.debug("Numeric-prefix upgrade: '%s' → '%s'", short_form, name)
        all_names -= short_to_remove

        text_len = len(full_text)
        if text_len > 1_000_000:
            min_freq = 10
        elif text_len > 200_000:
            min_freq = 5
        else:
            min_freq = 3

        entries: dict[str, EntityDictEntry] = {}
        for name in all_names:
            if name in _STOPWORDS:
                continue

            freq = max(word_freq.get(name, 0), ngram_only.get(name, 0))
            freq = max(
                freq, dialogue_names.get(name, 0), title_words.get(name, 0),
                naming_names.get(name, 0),
            )

            is_high_signal = name in dialogue_names or name in title_words or name in naming_names
            threshold = max(2, min_freq // 2) if is_high_signal else min_freq
            if freq < threshold:
                continue

            if name in naming_names:
                source, confidence = "naming", "high"
            elif name in dialogue_names:
                source, confidence = "dialogue", "high"
            elif name in title_words:
                source, confidence = "title", "high"
            elif name in suffix_types:
                source, confidence = "suffix", "medium"
            elif name in word_freq:
                source, confidence = "freq", "medium" if word_freq[name] >= 10 else "low"
            else:
                source, confidence = "ngram", "low"

            entity_type = suffix_types.get(name, "unknown")
            if (name in dialogue_names or name in naming_names) and entity_type == "unknown":
                entity_type = "person"

            entries[name] = EntityDictEntry(
                name=name,
                entity_type=entity_type,
                frequency=freq,
                confidence=confidence,
                aliases=[],
                source=source,
                sample_context=self._extract_sample_context(name, full_text),
            )

        sorted_entries = sorted(entries.values(), key=lambda e: e.frequency, reverse=True)
        top_entries = sorted_entries[:500]
        included = {e.name for e in top_entries}
        for entry in sorted_entries[500:]:
            if entry.name in naming_names and entry.name not in included:
                top_entries.append(entry)
        return top_entries

    @staticmethod
    def _dedup_ngrams(ngram_freq: Counter, jieba_names: set[str]) -> Counter:
        """Remove n-grams that are substrings of longer known words."""
        if not ngram_freq:
            return ngram_freq

        long_words = [w for w in (jieba_names | set(ngram_freq.keys())) if len(w) >= 3]
        substrings_to_remove: set[str] = set()
        for word in long_words:
            word_freq = max(ngram_freq.get(word, 0), 1)
            for start in range(len(word)):
                for end in range(start + 2, len(word)):
                    sub = word[start:end]
                    if sub in ngram_freq and sub != word:
                        if ngram_freq[sub] <= word_freq * 1.2:
                            substrings_to_remove.add(sub)

        return Counter({
            gram: freq for gram, freq in ngram_freq.items()
            if gram not in substrings_to_remove
        })

    def _extract_sample_context(
        self, name: str, full_text: str, context_len: int = 25
    ) -> str | None:
        """Extract a sample context around the first occurrence of a name."""
        idx = full_text.find(name)
        if idx < 0:
            return None
        start = max(0, idx - context_len)
        end = min(len(full_text), idx + len(name) + context_len)
        return full_text[start:end].replace("\n", " ").strip()

    # ── Phase 2: LLM classification ───────────────────────────────

    async def _classify_with_llm(
        self, candidates: list[EntityDictEntry]
    ) -> list[EntityDictEntry]:
        if not candidates or self.llm is None:
            return candidates

        system_prompt, user_prompt = build_prescan_prompt(candidates)
        result, _usage = await self.llm.generate(
            system=system_prompt,
            prompt=user_prompt,
            format={"type": "object"},
            temperature=0.1,
            max_tokens=8192,
        )

        if isinstance(result, str):
            import json

            result = json.loads(result)

        return self._merge_llm_results(candidates, result)

    def _merge_llm_results(
        self,
        candidates: list[EntityDictEntry],
        llm_result: dict,
    ) -> list[EntityDictEntry]:
        """Merge LLM classification results back into candidates."""
        by_name: dict[str, EntityDictEntry] = {c.name: c for c in candidates}

        for entity in llm_result.get("entities", []):
            name = entity.get("name", "")
            if name in by_name:
                if entity.get("type"):
                    by_name[name].entity_type = entity["type"]
                if entity.get("confidence"):
                    by_name[name].confidence = entity["confidence"]

        # Only keep alias-group members that were actually in the candidate list:
        # the LLM sometimes hallucinates extra members ("韩胖子" in 韩立's group).
        for group in llm_result.get("alias_groups", []):
            if not isinstance(group, list) or len(group) < 2:
                continue
            valid_group = [n for n in group if n in by_name]
            if len(valid_group) < 2:
                continue
            for name in valid_group:
                others = [n for n in valid_group if n != name]
                by_name[name].aliases = others

        rejected = set(llm_result.get("rejected", []))
        return [c for c in candidates if c.name not in rejected]
