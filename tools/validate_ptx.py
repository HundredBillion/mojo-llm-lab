import argparse
from pathlib import Path
import re


def validate_ptx(text):
    entries = re.findall(r'\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)', text, re.S)
    if len(entries) != 1 or '.extern' in text:
        raise ValueError('Require exactly one self-contained visible entry')
    entry, parameters = entries[0]
    declarations = parameters.split(',')
    names = []
    for declaration in declarations:
        match = re.fullmatch(r'\s*\.param\s+\.u64\s+\.ptr\s+\.align\s+1\s+(\w+)\s*', declaration)
        if match is None:
            raise ValueError('Require five flat 64-bit pointer parameters')
        names.append(match.group(1))
    if len(names) != 5 or len(set(names)) != 5:
        raise ValueError('Require five distinct pointer parameters')
    for directive, wanted in (('version', '8.1'), ('target', 'sm_86'), ('address_size', '64')):
        values = re.findall(r'^\.' + directive + r'\s+([^\n]+)$', text, re.M)
        if values != [wanted]:
            raise ValueError('Unsupported or duplicate PTX version, target or address width')
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ptx', type=Path)
    args = parser.parse_args()
    try:
        entry = validate_ptx(args.ptx.read_text())
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(2, str(error) + '\n')
    print(entry)


if __name__ == '__main__':
    main()
