import datetime
import regex as re


HTML_REGEX_NO_SPACE = r'(<[a-z][^>]+>|<\/?[a-z]>)'


def remove_tags_from_sent(str):
    return re.sub(HTML_REGEX_NO_SPACE, '', str)


def get_attr(tag, attr):
    return re.search(r'\s' + attr + r'=([^ ]+)', tag).group(1).strip('<>: ')


def sents_from_html(html_str):
    tps = html_str.split('<SEP>')
    return [tps[idx + 1] for idx, tp in enumerate(tps) if tp.startswith('<s') and idx + 1 < len(tps)]


def extract_sorted_notes_from_html(html, note_meta):
    notes = {}
    source_tps = html.split('<SEP>')
    for tp_idx, tp in enumerate(source_tps):
        if tp.startswith('<d'):
            note_id = get_attr(tp, 'note_id')
            note_end_idx = tp_idx + 1
            for note_end_idx in range(tp_idx + 1, len(source_tps)):
                if source_tps[note_end_idx] == '</d>':
                    break
            note_tps = source_tps[tp_idx:note_end_idx + 1]
            assert note_tps[-1] == '</d>'
            notes[note_id] = '<SEP>'.join(note_tps)
    note_id_order = [x['note_id'] for x in note_meta]
    try:
        notes_flat = [(k, v, note_id_order.index(k)) for k, v in notes.items()]
    except:
        print('Could not find note meta information.')
        assert len(note_meta) == 0
        notes_flat = [(k, v, get_date_from_note_id(k)) for k, v in notes.items()]
    notes_flat_sorted = list(sorted(notes_flat, key=lambda x: x[-1]))
    note_html = [x[1] for x in notes_flat_sorted]
    return '<SEP>'.join(note_html)


def get_date_from_note_id(note_id):
    x = note_id.split('-')
    year, month, day = int(x[2]), int(x[3]), int(x[4])
    hour = int(x[5])
    minute = int(x[6])
    obj = datetime.datetime(year, month, day, hour=hour, minute=minute)
    return obj


def transform_text(html_str, include_header=True, include_title=False):
    tps = html_str.split('<SEP>')
    curr_str = ''
    for idx, tp in enumerate(tps):
        if tp.startswith('<d') and include_title:
            curr_note_id = get_attr(tp, 'note_id')
            note_title = ' '.join(map(str.capitalize, curr_note_id.split('-')[9:]))
            curr_str += '\n\n<doc-sep>Title: ' + note_title + '\n'
        elif tp.startswith('<h'):
            raw = get_attr(tp, 'raw')
            if raw.lower() == 'unknown':
                continue
            raw_section = re.sub(r'[_\s]+', ' ', raw).strip()
            if len(curr_str) > 0 and curr_str[-1] != '\n':
                curr_str += '\n'
            if include_header:
                curr_str += raw_section + ':\n'
        elif idx > 0 and tps[idx - 1].startswith('<s'):
            sent_str = remove_tags_from_sent(tp)
            if len(curr_str) > 0 and curr_str[-1] not in {'\n', '\t', ' '}:
                curr_str += ' '
            curr_str += sent_str
    return curr_str.strip()


def split_into_sections(html_str):
    tps = html_str.split('<SEP>')
    sections = []
    curr_section_concept = ''
    curr_section_sents = []
    sent_idx_offset = 0
    for tp_idx, tp in enumerate(tps):
        if tp.startswith('<h'):
            curr_section_concept = get_attr(tp, 'con')
            curr_section_sents = []
        elif tp.startswith('<s'):
            # sent_body = remove_tags_from_sent(tps[tp_idx + 1].strip())
            sent_body = tps[tp_idx + 1].strip()
            curr_section_sents.append(sent_body)
        elif tp == '</h>':
            n = len(curr_section_sents)
            sections.append({
                'concept': curr_section_concept,
                'sents': curr_section_sents,
                'sent_idxs': list(range(sent_idx_offset, sent_idx_offset + n))
            })
            sent_idx_offset += n
    return sections