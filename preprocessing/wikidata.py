import sys
import ujson
from multiprocessing import Pool, cpu_count


def project(line, langs=('enwiki', 'ruwiki', 'ukwiki', 'eswiki')):
    if not line:
        return

    row = ujson.loads(line.strip())
    output = {lang: row['sitelinks'].get(lang, {}).get('title', '') for lang in langs}
    return ujson.dumps(output)


if __name__ == '__main__':
    src, dst = sys.argv[1:]

    with Pool(cpu_count()) as p, \
            open(src, 'r') as inp, \
            open(dst, 'w') as out:
        for output in p.imap(project, inp):
            if not output:
                continue

            out.write(output + '\n')
