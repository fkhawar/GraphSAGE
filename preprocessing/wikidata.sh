#!/usr/bin/env bash
lbzip2 -cd wikidata-20181112-all.json.bz2 | head -n -1 | tail -n +2 |sed 's/.$//' | jq -c .sitelinks > sitelinks
cat sitelinks | jq -c '[.[] | {(.site): .title}]' | jq -c add > titles