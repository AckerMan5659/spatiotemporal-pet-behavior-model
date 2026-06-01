import os
import re

# Species names treated as a "variant" dimension, not a sub-category.
# e.g. convulsion_cat_* and convulsion_dog_* are both grouped under 'convulsion'.
SPECIES = {'cat', 'dog', 'bird', 'rabbit'}

def get_group_key(name):
    """
    Determine the grouping key for a file/folder name.
    - normal_xxx (two segments)       -> 'normal'
    - normal_xxx_... (three+ segs)    -> 'normal_xxx'
    - YYYYMMDD_xxx_...                -> 'YYYYMMDD_xxx'
    - other_active_... / other_rest_  -> 'other_active' / 'other_rest'
      (second segment is a sub-category word, not a species)
    - convulsion_cat/dog_...          -> 'convulsion'
      (second segment is a species name, collapsed into first)
    - drink_0, eat_0, ...             -> 'drink', 'eat'
      (only two segments, second is numeric)
    - no underscore                   -> None (no grouping)
    """
    stem, ext = os.path.splitext(name)
    parts = stem.split('_')
    if len(parts) < 2:
        return None
    first = parts[0]

    # rule: first segment is 'normal'
    if first.lower() == 'normal':
        if len(parts) == 2:
            return 'normal'
        return 'normal_' + parts[1]

    # rule: first segment is a date-like number (e.g. 0501, 20250501)
    if re.fullmatch(r'\d{4,8}', first):
        return first + '_' + parts[1]

    # rule: second segment is a sub-category word (purely alphabetic, not a species)
    # e.g. other_active_0, other_rest_0 -> 'other_active', 'other_rest'
    if (len(parts) >= 3
            and re.fullmatch(r'[a-zA-Z]+', parts[1])
            and parts[1].lower() not in SPECIES):
        return first + '_' + parts[1]

    return first


def group_and_collapse(names, keep=1):
    """
    Group entries by computed key (see get_group_key).
    Groups with >1 member get collapsed; singletons stay as-is.
    Returns list of strings to display.
    """
    parsed = []
    for n in names:
        parsed.append((n, get_group_key(n)))

    # group consecutive entries with same key
    result = []
    i = 0
    while i < len(parsed):
        name, key = parsed[i]
        if key is None:
            result.append(name)
            i += 1
            continue
        # collect all consecutive with same key
        group = [parsed[i]]
        j = i + 1
        while j < len(parsed) and parsed[j][1] == key:
            group.append(parsed[j])
            j += 1
        if len(group) == 1:
            result.append(name)
        else:
            members = [g[0] for g in group]
            if len(members) <= keep * 2:
                result.extend(members)
            else:
                shown = members[:keep] + [f"... ({len(members) - keep*2} omitted) ..."] + members[-keep:]
                result.extend(shown)
        i = j
    return result


def print_label(path, prefix):
    """Print the contents of a label.txt inline, one line per entry."""
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]
    except OSError:
        return
    for line in lines:
        print(prefix + "    " + line)


def print_tree(path, prefix="", keep=1):
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]

    shown_dirs = group_and_collapse(dirs, keep)
    shown_files = group_and_collapse(files, keep)
    all_shown = shown_dirs + shown_files

    for i, name in enumerate(all_shown):
        is_last = (i == len(all_shown) - 1)
        connector = "└── " if is_last else "├── "
        print(prefix + connector + name)
        full = os.path.join(path, name)
        if os.path.isdir(full):
            extension = "    " if is_last else "│   "
            print_tree(full, prefix + extension, keep)
        elif name == "label.txt":
            print_label(full, prefix + ("    " if is_last else "│   "))


root = os.path.dirname(os.path.abspath(__file__))
print(root)
print_tree(root)
