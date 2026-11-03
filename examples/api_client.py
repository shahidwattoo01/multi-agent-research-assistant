"""Submit a job to the REST API and follow its Server-Sent Events stream.

research-assistant serve            # in another terminal
python examples/api_client.py "What is retrieval-augmented generation and when does it fail?"
"""

import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"


def main() -> None:
    question = (
        " ".join(sys.argv[1:]) or "What is retrieval-augmented generation and when does it fail?"
    )
    run_id = httpx.post(f"{BASE}/research", json={"question": question}).json()["run_id"]
    print(f"submitted {run_id}")

    with httpx.stream("GET", f"{BASE}/research/{run_id}/events", timeout=None) as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if "agent" in event:
                    print(f"[{event['agent']:>10}] {event['type']:<12} {event['message']}")

    print(httpx.get(f"{BASE}/research/{run_id}/report.md").text)


if __name__ == "__main__":
    main()
