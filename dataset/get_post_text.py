import csv
import json
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["text", "label"])

    for line in f_in:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)

        for post in data["data"]["children"]:
            d = post["data"]
            text = d.get("selftext", "").replace("\n", " ").strip()
            label = d.get("link_flair_text", "").strip()
            writer.writerow([text, label])