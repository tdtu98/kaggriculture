"""Build a standalone shop-adaptive agent from canonical route data."""

import argparse
import base64
import json
import zlib
from pathlib import Path


def encode_payload(payload):
    raw = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return base64.b85encode(zlib.compress(raw, 9)).decode("ascii")


def replace_payload(source, encoded):
    start = source.index("# BEGIN GENERATED ROUTES")
    end = source.index("# END GENERATED ROUTES")
    block = (
        "# BEGIN GENERATED ROUTES\n"
        "_PAYLOAD = json.loads(\n"
        f"    zlib.decompress(base64.b85decode({encoded!r})).decode('utf-8')\n"
        ")\n"
        "# END GENERATED ROUTES"
    )
    return source[:start] + block + source[end + len("# END GENERATED ROUTES") :]


def _submission_payload(routes):
    return {
        "schema_version": routes["schema_version"],
        "selector": routes["selector"],
        "branches": {
            name: {
                "source": branch["source"],
                "actions": branch["actions"],
                "weed_only": branch["weed_only"],
            }
            for name, branch in routes["branches"].items()
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    routes = json.loads(args.routes.read_text(encoding="utf-8"))
    source = args.template.read_text(encoding="utf-8")
    generated = replace_payload(
        source,
        encode_payload(_submission_payload(routes)),
    )
    args.output.write_text(generated, encoding="utf-8")


if __name__ == "__main__":
    main()
