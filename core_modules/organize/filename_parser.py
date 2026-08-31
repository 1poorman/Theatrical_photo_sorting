# -*- coding: utf-8 -*-
"""filename_parser - NCPA 剧照文件名/目录名解析器

从原始与整理完成两类命名中提取结构化元数据：
日期、剧场、剧目、事项（彩排/演出/幕次场景）、人物列表（演员饰角色）、摄影师、连拍序号。

已覆盖的命名变体（data/ncpa_test）：
  原始目录名:   【原始N】20170522歌剧院-歌剧《假面舞会》A组彩排、花絮-摄影牛小北
  原始文件名 1: 20170522-歌剧院-歌剧《假面舞会》A组彩排-芭芭拉·哈维曼饰阿米莉亚-凌风 (2).JPG  # 摄影师无前缀、连拍 (N)
  原始文件名 2: 20141012歌剧院-舞剧《马可·波罗》B演-前：苏鹏饰马可·波罗 (13)-摄影凌风.JPG     # 连拍 (N) 在人物后
  原始文件名 3: 20160609戏剧场-话剧《样式雷》 (10)-摄影凌风.JPG                                # 无人物段
  原始文件名 4: 20160609戏剧场-话剧《样式雷》-32-摄影凌风.JPG                                  # 连拍号 -N- 段
  原始文件名 5: 20140304戏剧场-京剧《天下归心》彩排-陈少云饰颍考叔-摄影肖翊06.jpg              # 摄影师名带数字后缀
  原始文件名 6: 20141012歌剧院-舞剧《马可·波罗》B演-李祎然饰中国公主摄影肖翊04.jpg             # 人物与摄影师无分隔符
  原始文件名 7: 20160609戏剧场-话剧《样式雷》彩排-左起：彭国斌饰雷廷昌杨淇饰慈禧-摄影凌风.JPG  # 无分隔符多人物
  整理完成:     03-20170522-歌剧院-歌剧《假面舞会》A组彩排 第1幕1场-左2起：王猛饰和恩、梅杰饰瑞宾-王小京.JPG
"""
import os
import re

# 常见剧场名（可扩展）
VENUES = ('歌剧院', '戏剧场', '音乐厅', '小剧场', '台湖剧场', '资料中心')


def _extract_people(seg):
    """从人物段提取（演员, 角色）对（位置算法，处理无分隔符连接）。

    规则：每个"饰"位置 i，
      actor = i 前最近的连续中文/·串（不含"饰"字）
      role  = i 后到下一"饰"的 actor 起点（或分隔符/段尾）之间的串
    无分隔符连接（如 彭国斌饰雷廷昌杨淇饰慈禧）时，对歧义串按
    「role 尽量长、下一 actor 2~4 字」启发式拆分。
    """
    seg = re.sub(r'^[^：]{0,6}：', '', seg)   # 去方位前缀（左起：/前右起：）

    def is_name_ch(c):
        return c == '·' or ('\u4e00' <= c <= '\u9fa5' and c != '饰')

    shi_pos = [m.start() for m in re.finditer('饰', seg)]
    # 预计算每个"饰"的 actor 起点（向前扫描，遇"饰"停止）
    actor_starts = []
    for sp in shi_pos:
        a_start = sp
        while a_start > 0 and is_name_ch(seg[a_start - 1]):
            a_start -= 1
        actor_starts.append(a_start)

    pairs = []
    for idx, sp in enumerate(shi_pos):
        actor = seg[actor_starts[idx]:sp].strip(' 、，,')
        # role 右边界：下一"饰"的 actor 起点（初始为最长连续串），或段尾
        end = actor_starts[idx + 1] if idx + 1 < len(shi_pos) else len(seg)
        role = re.split(r'[、，,（(]', seg[sp + 1:end])[0].strip(' 、，,')

        # 无分隔符歧义：两个"饰"之间为超长纯中文串（如 雷廷昌杨淇），
        # 按「role 尽量长、下一 actor 2~4 字」启发式重新拆分
        if idx + 1 < len(shi_pos):
            joint = seg[sp + 1:shi_pos[idx + 1]]
            if re.fullmatch(r'[\u4e00-\u9fa5·]+', joint) and len(joint) > 4:
                best = None
                for k in range(2, len(joint) - 1):
                    r, na = joint[:k], joint[k:]
                    if 2 <= len(r) <= 8 and 2 <= len(na) <= 4:
                        best = k
                if best:
                    role = joint[:best]
                    actor_starts[idx + 1] = sp + 1 + best

        if 2 <= len(actor) <= 12 and 2 <= len(role) <= 12:
            if (actor, role) not in pairs:
                pairs.append((actor, role))
    return pairs


# 连拍序号：(N)
BURST_PAREN_RE = re.compile(r'\((\d{1,3})\)')
# 摄影师 + 可选数字后缀：摄影肖翊06 / 摄影凌风
PHOTO_TAG_RE = re.compile(r'摄影[:：]?([\u4e00-\u9fa5]{2,3})\d{0,3}')
# 裸摄影师段：2~4 字中文人名（可带数字后缀）
PHOTOGRAPHER_RE = re.compile(r'^([\u4e00-\u9fa5]{2,4})\d{0,3}$')
# 日期 + 可选剧场（紧贴）
DATE_VENUE_RE = re.compile(r'^(\d{8})(' + '|'.join(VENUES) + r')?')
# 剧目名：类别前缀 + 《剧名》
PLAY_RE = re.compile(r'([\u4e00-\u9fa5]{0,4}《[^》]+》)')
# 整理完成文件名的序号前缀：NN-
SEQ_PREFIX_RE = re.compile(r'^(\d{1,3})-')


def _split_segments(stem):
    return [s.strip() for s in stem.split('-') if s.strip()]


def parse_stem(stem, done_style=False):
    """解析去掉扩展名的文件名主干。

    Returns:
        dict: {seq, date, venue, play, event, scene, people, photographer, burst_seq, base, raw}
    """
    info = {'seq': None, 'date': None, 'venue': None, 'play': None, 'event': None,
            'scene': None, 'people': [], 'photographer': None, 'burst_seq': None,
            'base': None, 'raw': stem}

    work = stem

    # 整理完成命名：序号前缀
    if done_style:
        m = SEQ_PREFIX_RE.match(work)
        if m:
            info['seq'] = int(m.group(1))
            work = work[m.end():]

    # 连拍序号：(N)（移除，避免干扰后续分段）
    m = BURST_PAREN_RE.search(work)
    if m:
        info['burst_seq'] = int(m.group(1))
        work = BURST_PAREN_RE.sub('', work)

    # 摄影师：优先匹配"摄影XX[NN]"（可能无分隔符直接接在人物后）
    m = PHOTO_TAG_RE.search(work)
    if m:
        info['photographer'] = m.group(1)
        work = work[:m.start()] + work[m.end():]
    else:
        # 裸人名段：按 '-' 分段后末段为 2~4 字中文
        segs_tmp = _split_segments(work)
        if len(segs_tmp) >= 2:
            m2 = PHOTOGRAPHER_RE.match(segs_tmp[-1])
            if m2 and '饰' not in segs_tmp[-1]:
                info['photographer'] = m2.group(1)
                work = work[:work.rfind(segs_tmp[-1])]

    work = work.strip().strip('-')
    segs = _split_segments(work)
    if not segs:
        return info

    # 连拍序号变体：独立纯数字段（《样式雷》-32- / 《马可·波罗》-10-摄影XX）
    if info['burst_seq'] is None and len(segs) >= 2 and segs[-1].isdigit():
        n = int(segs[-1])
        if 1 <= n <= 999:
            info['burst_seq'] = n
            segs = segs[:-1]

    # 连拍分组键：移除连拍标记与摄影师后的归一化主干
    info['base'] = ''.join(segs)

    # 日期 + 剧场：第一段（可能纯日期，剧场在第二段）
    m = DATE_VENUE_RE.match(segs[0])
    if m:
        info['date'] = m.group(1)
        info['venue'] = m.group(2)
        segs = segs[1:]
        if info['venue'] is None and segs and any(v in segs[0] for v in VENUES) and '饰' not in segs[0]:
            info['venue'] = segs[0]
            segs = segs[1:]

    # 剧目段：含《》；其后内容为事项/场景
    for i, seg in enumerate(segs):
        pm = PLAY_RE.search(seg)
        if pm:
            play = pm.group(1)
            rest = seg[pm.end():].strip()
            info['play'] = play
            if rest:
                # 整理完成命名中"剧名 场景"以空格分隔，场景含幕次信息
                if done_style and ' ' in rest:
                    event, scene = rest.split(' ', 1)
                    info['event'] = event or None
                    info['scene'] = scene or None
                else:
                    info['event'] = rest
            segs = segs[:i] + segs[i + 1:]
            break

    # 人物段：剩余段中含"饰"的段
    people = []
    seen = set()
    for seg in segs:
        for a, r in _extract_people(seg):
            if (a, r) not in seen:
                seen.add((a, r))
                people.append((a, r))
    info['people'] = people

    return info


def parse_filename(path, done_style=False):
    """解析图片文件路径/文件名。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return parse_stem(stem, done_style=done_style)


def parse_dirname(path):
    """解析原始摄影目录名，如：
    【原始2】20170522歌剧院-歌剧《假面舞会》A组彩排、花絮-摄影牛小北
    """
    name = os.path.basename(os.path.normpath(path))
    tag = None
    m = re.match(r'^【([^】]+)】(.*)$', name)
    if m:
        tag = m.group(1)
        name = m.group(2)

    info = {'tag': tag, 'date': None, 'venue': None, 'play': None,
            'event': None, 'photographer': None, 'raw': name}

    segs = _split_segments(name)
    if not segs:
        return info

    # 摄影师段（支持"摄影XX"或裸人名）
    if len(segs) >= 2:
        m = PHOTO_TAG_RE.search(segs[-1])
        if m:
            info['photographer'] = m.group(1)
            segs = segs[:-1]
        else:
            m2 = PHOTOGRAPHER_RE.match(segs[-1])
            if m2:
                info['photographer'] = m2.group(1)
                segs = segs[:-1]

    if not segs:
        return info

    # 日期 + 剧场
    m = DATE_VENUE_RE.match(segs[0])
    if m:
        info['date'] = m.group(1)
        info['venue'] = m.group(2)
        segs = segs[1:]
        if info['venue'] is None and segs and any(v in segs[0] for v in VENUES):
            info['venue'] = segs[0]
            segs = segs[1:]

    # 剧目 + 事项
    if segs:
        pm = PLAY_RE.search(segs[0])
        if pm:
            info['play'] = pm.group(1)
            rest = segs[0][pm.end():].strip()
            if rest:
                info['event'] = rest

    return info


def format_people(people):
    """[(actor, role)] -> 'X饰Y、Z饰W'"""
    return '、'.join(f'{a}饰{r}' for a, r in people)
