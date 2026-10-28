#!/data/data/com.termux/files/usr/bin/bash
# Termux:Boot hook. The installer replaces this file with the correct repository path.
if [[ -d "$HOME/purffle-shorts-cloud" ]]; then
  "$HOME/purffle-shorts-cloud/termux/purffle-service.sh" start
fi
