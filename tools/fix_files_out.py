# tools/fix_files_out.py
import sys, io


def fix(artifact: str) -> str:
    t = artifact.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("BEGIN_FILE "):
            i += 1
            continue
        path = line[11:].strip()
        i += 1
        body = []
        while i < len(lines) and lines[i].strip() != "END_FILE":
            body.append(lines[i])
            i += 1
        if i >= len(lines):
            raise SystemExit(f"Malformed artifact: missing END_FILE for {path}")
        i += 1  # skip END_FILE
        body_txt = "\n".join(body).replace("\r\n", "\n").replace("\r", "\n")
        # ENFORCE: body ends with exactly one LF
        body_txt = body_txt.rstrip("\n") + "\n"
        out.append(f"BEGIN_FILE {path}\n{body_txt}END_FILE\n")
    # ENFORCE: whole artifact ends with exactly one LF
    clean = "".join(out).rstrip("\n") + "\n"
    return clean


def main():
    if len(sys.argv) != 3:
        print("Usage: python tools/fix_files_out.py <in> <out>", file=sys.stderr)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    data = io.open(src, "r", encoding="utf-8", errors="replace").read()
    fixed = fix(data)
    io.open(dst, "w", encoding="utf-8", newline="\n").write(fixed)


if __name__ == "__main__":
    main()
