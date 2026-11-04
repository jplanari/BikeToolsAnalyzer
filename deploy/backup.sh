#!/bin/sh
# Nightly dump of the database, keeping the last 14. Run from the deploy folder, e.g. from cron:
#   15 3 * * *  cd /opt/biketools/deploy && ./backup.sh /var/backups/biketools
# Then copy the folder off this machine (rclone, rsync, a cloud bucket): a backup on the same disk is not a backup.
# Do NOT store BIKETOOLS_SECRET_KEY next to these files.
set -eu
DEST="${1:?usage: backup.sh <folder>}"
KEEP=14
mkdir -p "$DEST"
FILE="$DEST/biketools-$(date +%F-%H%M).dump"
docker compose exec -T db pg_dump -U biketools -d biketools --format=custom > "$FILE.partial"
mv "$FILE.partial" "$FILE"
ls -1t "$DEST"/biketools-*.dump | tail -n +$((KEEP + 1)) | xargs -r rm --
echo "saved $FILE"
