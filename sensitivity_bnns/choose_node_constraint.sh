#!/bin/bash

# Preferred node families or features
PREFERRED=("spr" "skylake" "cascadelake" "broadwell")

AVAILABLE=$(sinfo --Node --noheader --state=idle -o "%f" | tr ',' '\n' | sort | uniq)

for FEATURE in "${PREFERRED[@]}"; do
    if echo "$AVAILABLE" | grep -q "$FEATURE"; then
        echo "$FEATURE"
        exit 0
    fi
done

# No preferred nodes available
echo "none"
exit 0
